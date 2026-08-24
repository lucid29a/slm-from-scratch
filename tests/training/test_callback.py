"""Tests for callbacks (using the real Trainer with a synthetic dataset)."""

from __future__ import annotations

from pathlib import Path

from slm_from_scratch.tokenization.char import CharTokenizer, CharTokenizerConfig
from slm_from_scratch.training.callback import (
    Callback,
    CheckpointCallback,
    ConsoleLogger,
    LoggingCallback,
    SampleGenerationCallback,
    StepMetrics,
)
from slm_from_scratch.training.checkpoint import CheckpointManager
from slm_from_scratch.training.lr_schedule import CosineWithWarmup, LRScheduleConfig
from slm_from_scratch.training.optimizer import OptimizerConfig
from slm_from_scratch.training.trainer import Trainer, TrainerConfig
from tests.training.conftest import SyntheticLMDataset, make_tiny_model


class RecordingLogger:
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[dict[str, float], int]] = []
        self.text_calls: list[tuple[str, str, int]] = []

    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None:
        self.scalar_calls.append((scalars, step))

    def log_text(self, key: str, text: str, *, step: int) -> None:
        self.text_calls.append((key, text, step))


def make_trainer(**overrides: object) -> Trainer:
    model = make_tiny_model()
    dataset = SyntheticLMDataset(200, block_size=16, vocab_size=50)
    config_kwargs: dict[str, object] = {
        "max_steps": 6,
        "micro_batch_size": 4,
        "device": "cpu",
        "precision": "fp32",
        "seed": 1,
    }
    config_kwargs.update(overrides)
    config = TrainerConfig(**config_kwargs)  # type: ignore[arg-type]
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1e-3, warmup_steps=1, total_steps=6))
    return Trainer(
        model=model,
        train_dataset=dataset,
        config=config,
        optimizer_config=OptimizerConfig(),
        lr_schedule=schedule,
    )


def test_logging_callback_respects_log_every() -> None:
    logger = RecordingLogger()
    trainer = make_trainer()
    trainer.callbacks = [LoggingCallback(logger, log_every=2)]
    trainer.train()

    logged_steps = [step for _, step in logger.scalar_calls]
    assert logged_steps == [0, 2, 4]


def test_logging_callback_includes_loss_and_lr() -> None:
    logger = RecordingLogger()
    trainer = make_trainer()
    trainer.callbacks = [LoggingCallback(logger, log_every=1)]
    trainer.train()

    scalars, _ = logger.scalar_calls[0]
    assert "loss" in scalars
    assert "lr" in scalars


def test_console_logger_does_not_raise(capsys) -> None:
    logger = ConsoleLogger()
    logger.log_scalars({"loss": 1.23}, step=5)
    logger.log_text("sample", "hello world", step=5)
    captured = capsys.readouterr()
    assert "loss" in captured.out
    assert "hello world" in captured.out


def test_checkpoint_callback_saves_periodically(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    trainer = make_trainer(max_steps=6)
    trainer.checkpoint_manager = manager
    trainer.callbacks = [CheckpointCallback(manager, every_n_steps=3)]
    trainer.train()

    saved_steps = sorted(
        int(p.stem.split("_")[1]) for p in tmp_path.glob("step_*.pt")
    )
    # every_n_steps=3 over 6 steps -> saves at step 3 and 6; on_train_end also
    # saves a final checkpoint at the last completed step (6), so 6 may repeat.
    assert 3 in saved_steps
    assert 6 in saved_steps


def test_sample_generation_callback_produces_text_via_kv_cache() -> None:
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
    )
    callback = SampleGenerationCallback(tokenizer, prompt="the", max_new_tokens=8, every_n_steps=1)
    text = callback.generate(trainer)
    assert isinstance(text, str)
    assert len(text) >= len("the")
    assert model.training  # generate() restores train mode afterward


def test_custom_callback_hooks_are_called_in_order() -> None:
    calls: list[str] = []

    class TrackingCallback(Callback):
        def on_train_start(self, trainer: Trainer) -> None:
            calls.append("start")

        def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:
            calls.append(f"step_{metrics.step}")

        def on_train_end(self, trainer: Trainer) -> None:
            calls.append("end")

    trainer = make_trainer(max_steps=3)
    trainer.callbacks = [TrackingCallback()]
    trainer.train()

    assert calls == ["start", "step_0", "step_1", "step_2", "end"]
