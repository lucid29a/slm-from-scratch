"""Tests for Trainer, including the checkpoint-resume bit-exactness gate."""

from __future__ import annotations

from pathlib import Path

import torch

from slm_from_scratch.training.callback import Callback, StepMetrics
from slm_from_scratch.training.checkpoint import CheckpointManager
from slm_from_scratch.training.lr_schedule import CosineWithWarmup, LRScheduleConfig
from slm_from_scratch.training.optimizer import OptimizerConfig
from slm_from_scratch.training.trainer import Trainer, TrainerConfig
from tests.training.conftest import SyntheticLMDataset, make_tiny_model


class LossRecorder(Callback):
    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:
        self.losses.append(metrics.loss)


def build_trainer(
    dataset: SyntheticLMDataset,
    *,
    max_steps: int,
    checkpoint_manager: CheckpointManager | None = None,
) -> tuple[Trainer, LossRecorder]:
    model = make_tiny_model()
    config = TrainerConfig(
        max_steps=max_steps, micro_batch_size=4, device="cpu", precision="fp32", seed=123
    )
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1e-3, warmup_steps=2, total_steps=10))
    recorder = LossRecorder()
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        config=config,
        optimizer_config=OptimizerConfig(),
        lr_schedule=schedule,
        callbacks=[recorder],
        checkpoint_manager=checkpoint_manager,
    )
    return trainer, recorder


def test_train_runs_to_max_steps() -> None:
    dataset = SyntheticLMDataset(200, block_size=16, vocab_size=50)
    trainer, _ = build_trainer(dataset, max_steps=5)
    trainer.train()
    assert trainer.current_step == 5


def test_loss_decreases_over_enough_steps() -> None:
    dataset = SyntheticLMDataset(500, block_size=16, vocab_size=50)
    trainer, recorder = build_trainer(dataset, max_steps=50)
    trainer.train()
    early = sum(recorder.losses[:5]) / 5
    late = sum(recorder.losses[-5:]) / 5
    assert late < early


def test_gradient_accumulation_produces_same_step_count() -> None:
    dataset = SyntheticLMDataset(200, block_size=16, vocab_size=50)
    model = make_tiny_model()
    config = TrainerConfig(
        max_steps=4, micro_batch_size=2, grad_accum_steps=3, device="cpu", precision="fp32"
    )
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1e-3, total_steps=4))
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        config=config,
        optimizer_config=OptimizerConfig(),
        lr_schedule=schedule,
    )
    trainer.train()
    assert trainer.current_step == 4


def test_checkpoint_resume_reproduces_uninterrupted_loss_curve(tmp_path: Path) -> None:
    dataset = SyntheticLMDataset(300, block_size=16, vocab_size=50)

    # Uninterrupted: 10 steps straight.
    uninterrupted, uninterrupted_recorder = build_trainer(dataset, max_steps=10)
    uninterrupted.train()

    # Interrupted: 4 steps, checkpoint, then a *new* Trainer resumes to 10.
    manager = CheckpointManager(tmp_path)
    first_half, first_recorder = build_trainer(dataset, max_steps=4, checkpoint_manager=manager)
    first_half.train()
    manager.save(
        step=first_half.current_step, model=first_half.model, optimizer=first_half.optimizer
    )

    second_half, second_recorder = build_trainer(dataset, max_steps=10, checkpoint_manager=manager)
    assert second_half.current_step == 4  # resumed, not starting fresh
    second_half.train()

    resumed_losses = first_recorder.losses + second_recorder.losses
    assert len(resumed_losses) == len(uninterrupted_recorder.losses) == 10
    for a, b in zip(uninterrupted_recorder.losses, resumed_losses, strict=True):
        assert a == b  # bit-exact, not just close


def test_checkpoint_resume_reproduces_final_model_weights(tmp_path: Path) -> None:
    dataset = SyntheticLMDataset(300, block_size=16, vocab_size=50)

    uninterrupted, _ = build_trainer(dataset, max_steps=8)
    uninterrupted.train()

    manager = CheckpointManager(tmp_path)
    first_half, _ = build_trainer(dataset, max_steps=3, checkpoint_manager=manager)
    first_half.train()
    manager.save(
        step=first_half.current_step, model=first_half.model, optimizer=first_half.optimizer
    )

    second_half, _ = build_trainer(dataset, max_steps=8, checkpoint_manager=manager)
    second_half.train()

    pairs = zip(uninterrupted.model.parameters(), second_half.model.parameters(), strict=True)
    for p1, p2 in pairs:
        assert torch.equal(p1, p2)


def test_infer_block_size_matches_dataset() -> None:
    dataset = SyntheticLMDataset(200, block_size=16, vocab_size=50)
    trainer, _ = build_trainer(dataset, max_steps=1)
    assert trainer._infer_block_size() == 16
