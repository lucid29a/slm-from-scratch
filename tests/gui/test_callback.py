"""Tests for DashboardCallback and DashboardSampleCallback, via a real Trainer run."""

from __future__ import annotations

from slm_from_scratch.gui.callback import DashboardCallback, DashboardSampleCallback
from slm_from_scratch.gui.state import TrainingRunState
from slm_from_scratch.tokenization.char import CharTokenizer, CharTokenizerConfig
from slm_from_scratch.training.lr_schedule import CosineWithWarmup, LRScheduleConfig
from slm_from_scratch.training.optimizer import OptimizerConfig
from slm_from_scratch.training.trainer import Trainer, TrainerConfig
from tests.training.conftest import SyntheticLMDataset, make_tiny_model


def build_trainer_for_dashboard(state: TrainingRunState, *, max_steps: int) -> Trainer:
    dataset = SyntheticLMDataset(200, block_size=16, vocab_size=50)
    model = make_tiny_model()
    config = TrainerConfig(max_steps=max_steps, micro_batch_size=4, device="cpu", precision="fp32")
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1e-3, total_steps=max_steps))
    return Trainer(
        model=model,
        train_dataset=dataset,
        config=config,
        optimizer_config=OptimizerConfig(),
        lr_schedule=schedule,
        callbacks=[DashboardCallback(state)],
    )


def test_dashboard_callback_records_every_step() -> None:
    state = TrainingRunState()
    state.status = "running"
    trainer = build_trainer_for_dashboard(state, max_steps=5)
    trainer.train()

    history = state.snapshot_history()
    assert len(history) == 5
    assert [r.step for r in history] == [0, 1, 2, 3, 4]


def test_dashboard_callback_marks_finished_on_train_end() -> None:
    state = TrainingRunState()
    state.status = "running"
    trainer = build_trainer_for_dashboard(state, max_steps=3)
    trainer.train()
    # mypy narrows state.status to the "running" literal assigned above and
    # doesn't invalidate it across trainer.train() (which mutates it via the
    # DashboardCallback); confirmed correct at runtime.
    assert state.status == "finished"  # type: ignore[comparison-overlap]


def test_dashboard_callback_does_not_override_error_status() -> None:
    state = TrainingRunState()
    state.status = "error"
    trainer = build_trainer_for_dashboard(state, max_steps=2)
    trainer.train()
    # on_train_end only flips "running" -> "finished"; an externally-set error
    # status must survive.
    assert state.status == "error"


def test_dashboard_sample_callback_stores_generated_text_in_state() -> None:
    state = TrainingRunState()
    tokenizer = CharTokenizer(CharTokenizerConfig(vocab_size=40)).train(
        ["the quick brown fox jumps over the lazy dog"]
    )
    model = make_tiny_model(vocab_size=tokenizer.vocab_size, block_size=32)
    dataset = SyntheticLMDataset(100, block_size=16, vocab_size=tokenizer.vocab_size)
    config = TrainerConfig(max_steps=2, micro_batch_size=2, device="cpu", precision="fp32")
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1e-3, total_steps=2))
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        config=config,
        optimizer_config=OptimizerConfig(),
        lr_schedule=schedule,
        callbacks=[
            DashboardSampleCallback(
                state, tokenizer, prompt="the", max_new_tokens=5, every_n_steps=1
            )
        ],
    )
    trainer.train()
    assert state.latest_sample != ""
