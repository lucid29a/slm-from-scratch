"""Tests for the Gradio app: model loading and chat response generation.

Full closures (start/stop/poll training) live inside build_app() and are
exercised via the callback/state tests plus a manual end-to-end run; here we
test the two pure, importable pieces: app construction and chat.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pytest
import yaml

from slm_from_scratch.gui.app import _chat_respond, _load_chat_model, build_app
from slm_from_scratch.tokenization.byte_level_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerConfig,
)
from slm_from_scratch.training.checkpoint import CheckpointManager
from tests.training.conftest import make_tiny_model


def test_build_app_returns_blocks() -> None:
    app = build_app()
    assert isinstance(app, gr.Blocks)


@pytest.fixture
def trained_checkpoint(tmp_path: Path) -> tuple[Path, Path, Path]:
    tokenizer = ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=300)).train(
        ["the quick brown fox jumps over the lazy dog"]
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    model = make_tiny_model(vocab_size=tokenizer.vocab_size, block_size=32)
    checkpoint_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir)
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager.save(step=1, model=model, optimizer=optimizer)

    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocab_size": tokenizer.vocab_size,
                    "n_layer": 2,
                    "n_head": 4,
                    "n_embd": 32,
                    "block_size": 32,
                    "dropout": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    return model_config_path, checkpoint_dir, tokenizer_path


def test_load_chat_model_succeeds(trained_checkpoint: tuple[Path, Path, Path]) -> None:
    model_config_path, checkpoint_dir, tokenizer_path = trained_checkpoint
    bundle, status = _load_chat_model(
        str(model_config_path), str(checkpoint_dir), str(tokenizer_path)
    )
    assert bundle is not None
    assert "Loaded" in status


def test_load_chat_model_missing_checkpoint_reports_error(
    trained_checkpoint: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    model_config_path, _checkpoint_dir, tokenizer_path = trained_checkpoint
    empty_dir = tmp_path / "empty_checkpoints"
    empty_dir.mkdir()
    bundle, status = _load_chat_model(str(model_config_path), str(empty_dir), str(tokenizer_path))
    assert bundle is None
    assert "No checkpoint" in status


def test_load_chat_model_bad_config_reports_error(tmp_path: Path) -> None:
    bundle, status = _load_chat_model(str(tmp_path / "nope.yaml"), str(tmp_path), str(tmp_path))
    assert bundle is None
    assert "Failed to load" in status


def test_chat_respond_without_model_returns_placeholder() -> None:
    history, cleared = _chat_respond("hello", [], None, 0.8, 40, 1.0, 20)
    assert history[-1]["role"] == "assistant"
    assert "No model loaded" in history[-1]["content"]
    assert cleared == ""


def test_chat_respond_with_model_generates_a_reply(
    trained_checkpoint: tuple[Path, Path, Path],
) -> None:
    model_config_path, checkpoint_dir, tokenizer_path = trained_checkpoint
    bundle, _status = _load_chat_model(
        str(model_config_path), str(checkpoint_dir), str(tokenizer_path)
    )
    assert bundle is not None

    history, cleared = _chat_respond("the quick", [], bundle, 0.8, 40, 1.0, 10)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "the quick"}
    assert history[1]["role"] == "assistant"
    assert isinstance(history[1]["content"], str)
    assert cleared == ""


def test_chat_respond_uses_prior_turns_as_context(
    trained_checkpoint: tuple[Path, Path, Path],
) -> None:
    model_config_path, checkpoint_dir, tokenizer_path = trained_checkpoint
    bundle, _status = _load_chat_model(
        str(model_config_path), str(checkpoint_dir), str(tokenizer_path)
    )
    assert bundle is not None

    prior_history = [
        {"role": "user", "content": "the quick brown"},
        {"role": "assistant", "content": "fox jumps"},
    ]
    history, _cleared = _chat_respond("over the", prior_history, bundle, 0.8, 40, 1.0, 5)
    assert history[0] == prior_history[0]
    assert history[1] == prior_history[1]
    assert history[2] == {"role": "user", "content": "over the"}
