"""The training loop: owns the loop and nothing else.

Everything that isn't strictly "run forward, run backward, step the optimizer"
is delegated: batch sampling to the dataset directly (via random-offset draws,
the standard LLM-pretraining recipe), optimizer construction to
:class:`~slm_from_scratch.training.optimizer.OptimizerFactory`, the LR curve to
an :class:`~slm_from_scratch.training.lr_schedule.LRSchedule`, mixed precision
to :class:`~slm_from_scratch.training.precision.PrecisionPolicy`, gradient
accumulation/clipping to their own small objects, and every observable side
effect (logging, checkpointing, eval, sampling) to
:class:`~slm_from_scratch.training.callback.Callback` instances.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset

from slm_from_scratch.core.config import BaseConfig
from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.training.callback import Callback, StepMetrics
from slm_from_scratch.training.checkpoint import CheckpointManager
from slm_from_scratch.training.distributed import DistributedStrategy, SingleDeviceStrategy
from slm_from_scratch.training.gradient import GradientAccumulator, GradientClipper
from slm_from_scratch.training.lr_schedule import LRSchedule
from slm_from_scratch.training.optimizer import OptimizerConfig, OptimizerFactory
from slm_from_scratch.training.precision import PrecisionPolicy

__all__ = ["Trainer", "TrainerConfig"]


@dataclass(frozen=True, kw_only=True)
class TrainerConfig(BaseConfig):
    """Configuration for :class:`Trainer`.

    Attributes:
        max_steps: Total number of optimizer steps to run.
        micro_batch_size: Sequences per forward/backward pass.
        grad_accum_steps: Micro-batches accumulated per optimizer step. The
            effective batch size is ``micro_batch_size * grad_accum_steps``.
        grad_clip: Maximum gradient norm; ``None`` disables clipping.
        precision: Compute precision -- ``"fp32"``, ``"fp16"``, or ``"bf16"``.
        seed: Seed for the batch-sampling RNG (and torch's global RNG at start).
        device: Device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    max_steps: int
    micro_batch_size: int
    grad_accum_steps: int = 1
    grad_clip: float | None = 1.0
    precision: str = "bf16"
    seed: int = 1337
    device: str = "cuda"

    def validate(self) -> None:
        """Check batch/step sizes are positive."""
        if self.max_steps <= 0:
            raise ConfigError(f"max_steps must be positive, got {self.max_steps}")
        if self.micro_batch_size <= 0:
            raise ConfigError(f"micro_batch_size must be positive, got {self.micro_batch_size}")
        if self.grad_accum_steps <= 0:
            raise ConfigError(f"grad_accum_steps must be positive, got {self.grad_accum_steps}")


class Trainer:
    """Runs the training loop for a language model.

    Args:
        model: The model to train (moved to ``config.device`` internally).
        train_dataset: A dataset yielding ``(input_ids, target_ids)`` pairs;
            sampled with replacement at random offsets each micro-batch.
        config: Loop-level configuration (steps, batch size, precision, ...).
        optimizer_config: AdamW hyperparameters.
        lr_schedule: Step -> learning-rate function.
        callbacks: Hooks run at training start/step-end/train-end.
        distributed: How to talk to one or many devices; defaults to
            :class:`~slm_from_scratch.training.distributed.SingleDeviceStrategy`.
        checkpoint_manager: If given and a checkpoint already exists in its
            directory, training resumes from the latest one automatically.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
        config: TrainerConfig,
        optimizer_config: OptimizerConfig,
        lr_schedule: LRSchedule,
        callbacks: list[Callback] | None = None,
        distributed: DistributedStrategy | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.config = config
        self.distributed = distributed or SingleDeviceStrategy()
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )

        self.model = self.distributed.wrap_model(model.to(self.device))
        self.train_dataset = train_dataset
        self.lr_schedule = lr_schedule
        self.callbacks = callbacks or []
        self.checkpoint_manager = checkpoint_manager

        self.optimizer = OptimizerFactory(optimizer_config).build(
            self.distributed.unwrap_model(self.model)
        )
        self.precision = PrecisionPolicy(config.precision, device_type=self.device.type)
        self.accumulator = GradientAccumulator(config.grad_accum_steps)
        self.clipper = GradientClipper(config.grad_clip)

        # Batch sampling draws from torch's global RNG (not a private Generator)
        # specifically so CheckpointManager's saved `torch.get_rng_state()`
        # captures it -- that's what makes a resumed run's batch sequence, not
        # just its weights, pick up exactly where an uninterrupted run would be.
        torch.manual_seed(config.seed)

        self.current_step = 0
        self._tokens_seen = 0
        if self.checkpoint_manager is not None:
            self._maybe_resume()

    def train(self) -> None:
        """Run the training loop from :attr:`current_step` to ``config.max_steps``."""
        for cb in self.callbacks:
            cb.on_train_start(self)

        block_size = self._infer_block_size()
        tokens_per_step = (
            self.config.micro_batch_size * self.config.grad_accum_steps * block_size
        )

        while self.current_step < self.config.max_steps:
            lr = self.lr_schedule.lr_at(self.current_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            total_loss = 0.0
            grad_norm: float | None = None
            for _ in range(self.config.grad_accum_steps):
                inputs, targets = self._sample_batch()
                with self.precision.autocast():
                    _, loss = self.model(inputs, targets)
                assert loss is not None
                total_loss += loss.item()
                scaled = self.accumulator.scale_loss(loss)
                scaled.backward()  # type: ignore[no-untyped-call]  # torch stub gap, not ours

            grad_norm = self.clipper.clip(self.distributed.unwrap_model(self.model))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self._tokens_seen += tokens_per_step
            metrics = StepMetrics(
                step=self.current_step,
                loss=total_loss / self.config.grad_accum_steps,
                learning_rate=lr,
                grad_norm=grad_norm,
                tokens_seen=self._tokens_seen,
            )
            for cb in self.callbacks:
                cb.on_step_end(self, metrics)

            self.current_step += 1

        for cb in self.callbacks:
            cb.on_train_end(self)

    def _sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a random-offset micro-batch (with replacement) from the training dataset."""
        n = len(self.train_dataset)  # type: ignore[arg-type]
        indices = torch.randint(0, n, (self.config.micro_batch_size,))
        inputs, targets = zip(
            *(self.train_dataset[int(i)] for i in indices), strict=True
        )
        return (
            torch.stack(inputs).to(self.device, non_blocking=True),
            torch.stack(targets).to(self.device, non_blocking=True),
        )

    def _infer_block_size(self) -> int:
        """Infer the sequence length from one sample of the training dataset."""
        sample_input, _ = self.train_dataset[0]
        return int(sample_input.shape[0])

    def _maybe_resume(self) -> None:
        assert self.checkpoint_manager is not None
        state = self.checkpoint_manager.load_latest()
        if state is None:
            return
        CheckpointManager.restore(
            state, model=self.distributed.unwrap_model(self.model), optimizer=self.optimizer
        )
        self.current_step = state.step
