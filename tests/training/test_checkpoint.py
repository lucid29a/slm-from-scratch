"""Tests for CheckpointManager."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from slm_from_scratch.core.exceptions import CheckpointError
from slm_from_scratch.training.checkpoint import CheckpointManager


def make_model_and_optimizer() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_save_creates_a_file(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = make_model_and_optimizer()
    path = manager.save(step=10, model=model, optimizer=optimizer)
    assert path.is_file()


def test_load_round_trips_model_weights(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = make_model_and_optimizer()
    manager.save(step=5, model=model, optimizer=optimizer)

    new_model, new_optimizer = make_model_and_optimizer()
    state = manager.load_latest()
    assert state is not None
    CheckpointManager.restore(state, model=new_model, optimizer=new_optimizer)

    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_load_latest_returns_none_when_empty(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    assert manager.load_latest() is None


def test_load_latest_picks_highest_step(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = make_model_and_optimizer()
    manager.save(step=1, model=model, optimizer=optimizer)
    manager.save(step=100, model=model, optimizer=optimizer)
    manager.save(step=50, model=model, optimizer=optimizer)

    state = manager.load_latest()
    assert state is not None
    assert state.step == 100


def test_load_missing_file_raises(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    with pytest.raises(CheckpointError, match="not found"):
        manager.load(tmp_path / "nope.pt")


def test_restore_resets_rng_state(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = make_model_and_optimizer()

    torch.manual_seed(0)
    _ = torch.randn(5)  # advance the RNG
    manager.save(step=1, model=model, optimizer=optimizer)
    draw_after_save = torch.randn(5)

    # Advance the RNG further, then restore -- the next draw should match
    # draw_after_save exactly, since restore() resets to the saved RNG state.
    _ = torch.randn(100)
    state = manager.load_latest()
    assert state is not None
    CheckpointManager.restore(state, model=model, optimizer=optimizer)
    draw_restored = torch.randn(5)

    assert torch.equal(draw_after_save, draw_restored)


def test_optimizer_state_round_trips(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = make_model_and_optimizer()

    # Take a step so AdamW has real moment state to save.
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()

    manager.save(step=1, model=model, optimizer=optimizer)

    new_model, new_optimizer = make_model_and_optimizer()
    # New optimizer needs at least one step to allocate state before restoring into it.
    new_model(torch.randn(2, 4)).sum().backward()
    new_optimizer.step()

    state = manager.load_latest()
    assert state is not None
    CheckpointManager.restore(state, model=new_model, optimizer=new_optimizer)

    orig_exp_avg = optimizer.state[optimizer.param_groups[0]["params"][0]]["exp_avg"]
    new_exp_avg = new_optimizer.state[new_optimizer.param_groups[0]["params"][0]]["exp_avg"]
    assert torch.equal(orig_exp_avg, new_exp_avg)
