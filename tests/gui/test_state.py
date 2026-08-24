"""Tests for TrainingRunState."""

from __future__ import annotations

from slm_from_scratch.gui.state import TrainingRunState


def test_starts_idle_with_empty_history() -> None:
    state = TrainingRunState()
    assert state.status == "idle"
    assert state.snapshot_history() == []


def test_record_step_appends_to_history() -> None:
    state = TrainingRunState()
    state.record_step(step=0, loss=2.5, learning_rate=1e-3)
    state.record_step(step=1, loss=2.3, learning_rate=1e-3)
    history = state.snapshot_history()
    assert len(history) == 2
    assert history[0].step == 0
    assert history[1].loss == 2.3


def test_snapshot_history_is_a_copy() -> None:
    state = TrainingRunState()
    state.record_step(step=0, loss=1.0, learning_rate=1e-3)
    snapshot = state.snapshot_history()
    state.record_step(step=1, loss=0.9, learning_rate=1e-3)
    assert len(snapshot) == 1  # the earlier snapshot is unaffected by the later append


def test_reset_clears_everything() -> None:
    state = TrainingRunState()
    state.status = "finished"
    state.record_step(step=0, loss=1.0, learning_rate=1e-3)
    state.latest_sample = "hello"
    state.error = "oops"
    old_stop_event = state.stop_event
    old_stop_event.set()

    state.reset()

    # mypy narrows `state.status` to the literal assigned above and doesn't
    # invalidate that narrowing across the reset() call; these two lines are
    # confirmed correct at runtime (see the passing test), not a real bug.
    assert state.status == "idle"  # type: ignore[comparison-overlap]
    assert state.snapshot_history() == []  # type: ignore[unreachable]
    assert state.latest_sample == ""
    assert state.error is None
    assert state.stop_event is not old_stop_event
    assert not state.stop_event.is_set()
