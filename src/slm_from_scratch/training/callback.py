"""Training callbacks: behavior injected into the loop, not hard-coded into it.

:class:`~slm_from_scratch.training.trainer.Trainer` owns the loop and nothing
else -- logging, checkpointing, periodic evaluation, sample generation, and
throughput reporting are all :class:`Callback` implementations the trainer
calls at fixed points. Adding a new kind of monitoring means writing a new
callback, never editing the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from slm_from_scratch.core.protocols import Loggable

if TYPE_CHECKING:
    from slm_from_scratch.training.checkpoint import CheckpointManager
    from slm_from_scratch.training.trainer import Trainer

__all__ = [
    "Callback",
    "CheckpointCallback",
    "ConsoleLogger",
    "EvalCallback",
    "LoggingCallback",
    "SampleGenerationCallback",
    "StepMetrics",
    "TensorBoardLogger",
    "ThroughputCallback",
    "WandbLogger",
]


@dataclass
class StepMetrics:
    """What a :class:`Callback` learns about a just-completed training step.

    Attributes:
        step: The step index that just completed.
        loss: The (unscaled) training loss for this step.
        learning_rate: The learning rate used for this step.
        grad_norm: Pre-clip gradient norm, if gradient clipping is enabled.
        tokens_seen: Total tokens processed so far, across the whole run.
        extra: Additional callback-specific values.
    """

    step: int
    loss: float
    learning_rate: float
    grad_norm: float | None = None
    tokens_seen: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class Callback:
    """Base class for training-loop hooks. Every method is an optional no-op override.

    Not an ABC: a callback that only cares about, say, step-end has no reason
    to be forced to override the other two hooks.
    """

    def on_train_start(self, trainer: Trainer) -> None:
        """Called once, before the first training step."""

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Called after each completed optimizer step."""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once, after the last training step."""


# --------------------------------------------------------------------------- #
# Loggers (implement the Loggable protocol from core.protocols)
# --------------------------------------------------------------------------- #
class ConsoleLogger:
    """Prints scalars and text samples to stdout."""

    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None:
        """Print one line: ``step N | key=value key=value ...``."""
        formatted = " ".join(f"{k}={v:.4g}" for k, v in scalars.items())
        print(f"step {step:>8d} | {formatted}")

    def log_text(self, key: str, text: str, *, step: int) -> None:
        """Print a labeled text block."""
        print(f"step {step:>8d} | {key}:\n{text}")


class TensorBoardLogger:
    """Logs to a TensorBoard event file via ``torch.utils.tensorboard``.

    Requires the optional ``logging`` extra (``pip install -e ".[logging]"``).
    """

    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None:
        """Write each scalar to its own TensorBoard tag."""
        for key, value in scalars.items():
            self._writer.add_scalar(key, value, global_step=step)

    def log_text(self, key: str, text: str, *, step: int) -> None:
        """Write a text sample under ``key``."""
        self._writer.add_text(key, text, global_step=step)


class WandbLogger:
    """Logs to a Weights & Biases run.

    Requires the optional ``logging`` extra and an authenticated ``wandb`` session.
    """

    def __init__(self, **init_kwargs: Any) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)

    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None:
        """Log a batch of scalars at ``step``."""
        self._run.log(scalars, step=step)

    def log_text(self, key: str, text: str, *, step: int) -> None:
        """Log a text sample under ``key``."""
        self._run.log({key: self._wandb.Html(f"<pre>{text}</pre>")}, step=step)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
class LoggingCallback(Callback):
    """Logs step metrics to a :class:`~slm_from_scratch.core.protocols.Loggable` sink.

    Args:
        logger: Where to send scalars (console, TensorBoard, W&B, ...).
        log_every: Log every ``log_every`` steps (always logs step 0).
    """

    def __init__(self, logger: Loggable, *, log_every: int = 10) -> None:
        self.logger = logger
        self.log_every = log_every

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:  # noqa: ARG002
        """Log ``loss``/``lr``/``grad_norm`` (and any extras) every ``log_every`` steps."""
        if metrics.step % self.log_every != 0:
            return
        scalars = {"loss": metrics.loss, "lr": metrics.learning_rate}
        if metrics.grad_norm is not None:
            scalars["grad_norm"] = metrics.grad_norm
        scalars.update(metrics.extra)
        self.logger.log_scalars(scalars, step=metrics.step)


class CheckpointCallback(Callback):
    """Periodically saves training state via a :class:`CheckpointManager`.

    Args:
        checkpoint_manager: Where and how to save.
        every_n_steps: Save every this many steps.
    """

    def __init__(self, checkpoint_manager: CheckpointManager, *, every_n_steps: int) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.every_n_steps = every_n_steps

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Save a checkpoint every ``every_n_steps`` steps (step count is 1-indexed here)."""
        if (metrics.step + 1) % self.every_n_steps == 0:
            self.checkpoint_manager.save(
                step=metrics.step + 1, model=trainer.model, optimizer=trainer.optimizer
            )

    def on_train_end(self, trainer: Trainer) -> None:
        """Always save a final checkpoint at the end of training."""
        self.checkpoint_manager.save(
            step=trainer.current_step, model=trainer.model, optimizer=trainer.optimizer
        )


class ThroughputCallback(Callback):
    """Reports tokens/second and (if peak FLOPs are known) model FLOPs utilization.

    MFU is estimated with the standard ``6 * N * tokens`` approximation (Chinchilla,
    PaLM) for forward+backward FLOPs of a dense transformer, where ``N`` is the
    non-embedding parameter count.

    Args:
        num_params: Non-embedding parameter count (see
            :meth:`~slm_from_scratch.modeling.base.LanguageModel.num_parameters`).
        tokens_per_step: Number of tokens processed per optimizer step.
        peak_flops: This device's peak FLOPs/s at the training precision, if known.
        log_every: Report every this many steps.
    """

    def __init__(
        self,
        *,
        num_params: int,
        tokens_per_step: int,
        peak_flops: float | None = None,
        log_every: int = 50,
    ) -> None:
        self.num_params = num_params
        self.tokens_per_step = tokens_per_step
        self.peak_flops = peak_flops
        self.log_every = log_every
        self._last_time: float | None = None
        self._last_step: int = -1

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:  # noqa: ARG002
        """Compute and log throughput since the last report."""
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            self._last_step = metrics.step
            return
        if metrics.step % self.log_every != 0:
            return

        elapsed = now - self._last_time
        steps_elapsed = metrics.step - self._last_step
        tokens = steps_elapsed * self.tokens_per_step
        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0.0

        report = {"tokens_per_sec": tokens_per_sec}
        if self.peak_flops:
            achieved_flops = 6 * self.num_params * tokens_per_sec
            report["mfu"] = achieved_flops / self.peak_flops
        formatted = " ".join(f"{k}={v:.4g}" for k, v in report.items())
        print(f"step {metrics.step:>8d} | throughput: {formatted}")

        self._last_time = now
        self._last_step = metrics.step


class EvalCallback(Callback):
    """Periodically computes average loss on a held-out dataset.

    Args:
        val_dataset: A dataset yielding ``(input_ids, target_ids)`` pairs.
        batch_size: Micro-batch size for evaluation.
        every_n_steps: Evaluate every this many steps.
        max_batches: Cap the number of batches evaluated, for speed.
        logger: Optional sink to also log the eval loss to.
    """

    def __init__(
        self,
        val_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
        *,
        batch_size: int,
        every_n_steps: int,
        max_batches: int = 50,
        logger: Loggable | None = None,
    ) -> None:
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches
        self.logger = logger

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Evaluate and (optionally) log every ``every_n_steps`` steps."""
        if metrics.step == 0 or metrics.step % self.every_n_steps != 0:
            return
        eval_loss = self.evaluate(trainer)
        print(f"step {metrics.step:>8d} | eval_loss={eval_loss:.4g}")
        if self.logger is not None:
            self.logger.log_scalars({"eval_loss": eval_loss}, step=metrics.step)

    @torch.no_grad()
    def evaluate(self, trainer: Trainer) -> float:
        """Compute mean loss over up to ``max_batches`` batches of ``val_dataset``."""
        model = trainer.model
        was_training = model.training
        model.eval()

        total_loss, total_batches = 0.0, 0
        n = len(self.val_dataset)  # type: ignore[arg-type]
        for start in range(0, min(n, self.max_batches * self.batch_size), self.batch_size):
            indices = range(start, min(start + self.batch_size, n))
            inputs = torch.stack([self.val_dataset[i][0] for i in indices]).to(trainer.device)
            targets = torch.stack([self.val_dataset[i][1] for i in indices]).to(trainer.device)
            with trainer.precision.autocast():
                _, loss = model(inputs, targets)
            assert loss is not None
            total_loss += loss.item()
            total_batches += 1

        if was_training:
            model.train()
        return total_loss / max(total_batches, 1)


class SampleGenerationCallback(Callback):
    """Periodically generates a short greedy sample, to eyeball training progress.

    A minimal, self-contained generation loop (greedy, KV-cached) -- not the
    full sampling API in :mod:`slm_from_scratch.inference`, which is built
    later and used for actual text generation.

    Args:
        tokenizer: Tokenizer to encode the prompt and decode the sample.
        prompt: Text prompt to continue from.
        max_new_tokens: Number of tokens to generate.
        every_n_steps: Generate every this many steps.
        logger: Optional sink to also log the sample text to.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt: str,
        max_new_tokens: int = 64,
        every_n_steps: int,
        logger: Loggable | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.every_n_steps = every_n_steps
        self.logger = logger

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:
        """Generate and print/log a sample every ``every_n_steps`` steps."""
        if metrics.step == 0 or metrics.step % self.every_n_steps != 0:
            return
        text = self.generate(trainer)
        print(f"step {metrics.step:>8d} | sample:\n{text}")
        if self.logger is not None:
            self.logger.log_text("sample", text, step=metrics.step)

    @torch.no_grad()
    def generate(self, trainer: Trainer) -> str:
        """Greedily generate :attr:`max_new_tokens` tokens continuing :attr:`prompt`."""
        model = trainer.model
        was_training = model.training
        model.eval()

        ids = self.tokenizer.encode(self.prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=trainer.device)

        new_kv_cache = getattr(model, "new_kv_cache", None)
        kv_cache = new_kv_cache() if callable(new_kv_cache) else None
        generated = list(ids)
        current = input_ids
        for _ in range(self.max_new_tokens):
            logits, _ = model(current, kv_cache=kv_cache)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            if next_id == self.tokenizer.eos_id:
                break
            current = torch.tensor([[next_id]], dtype=torch.long, device=trainer.device)
            if kv_cache is None:
                current = torch.tensor([generated], dtype=torch.long, device=trainer.device)

        if was_training:
            model.train()
        return str(self.tokenizer.decode(generated))
