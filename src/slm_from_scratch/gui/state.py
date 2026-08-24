"""Shared, thread-safe state for a training run launched from the GUI.

A run's :class:`Trainer` executes on a background thread (Gradio handlers run
per-request and must return promptly); this is the object that thread writes
to and the GUI's polling timer reads from. Every mutation is a single
attribute assignment or list append under the lock -- cheap enough that the
lock is never a bottleneck against a training step's actual cost.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["StepRecord", "TrainingRunState"]

RunStatus = Literal["idle", "running", "stopping", "finished", "stopped", "error"]


@dataclass
class StepRecord:
    """One training step's metrics, as recorded for the dashboard's loss plot."""

    step: int
    loss: float
    learning_rate: float


@dataclass
class TrainingRunState:
    """Mutable state shared between a background training thread and the GUI.

    Attributes:
        status: Current run status.
        history: One :class:`StepRecord` per completed step, in order.
        latest_sample: The most recent generated sample, if any.
        error: The exception message, if ``status == "error"``.
        stop_event: Set to request the run stop after its current step.
    """

    status: RunStatus = "idle"
    history: list[StepRecord] = field(default_factory=list)
    latest_sample: str = ""
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        """Reset to a fresh, idle state (e.g. before starting a new run)."""
        with self._lock:
            self.status = "idle"
            self.history = []
            self.latest_sample = ""
            self.error = None
            self.stop_event = threading.Event()

    def record_step(self, *, step: int, loss: float, learning_rate: float) -> None:
        """Append one step's metrics to :attr:`history`."""
        with self._lock:
            self.history.append(StepRecord(step=step, loss=loss, learning_rate=learning_rate))

    def snapshot_history(self) -> list[StepRecord]:
        """Return a shallow copy of :attr:`history`, safe to read without racing appends."""
        with self._lock:
            return list(self.history)
