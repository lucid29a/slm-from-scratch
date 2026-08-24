"""Callbacks that stream a running Trainer's progress into a TrainingRunState.

The state object is what :mod:`slm_from_scratch.gui.app`'s polling timer
reads from to update the dashboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from slm_from_scratch.gui.state import TrainingRunState
from slm_from_scratch.training.callback import Callback, SampleGenerationCallback, StepMetrics

if TYPE_CHECKING:
    from slm_from_scratch.training.trainer import Trainer

__all__ = ["DashboardCallback", "DashboardSampleCallback"]


class DashboardCallback(Callback):
    """Records every step's loss/LR into a :class:`TrainingRunState` for the live plot."""

    def __init__(self, state: TrainingRunState) -> None:
        self.state = state

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:  # noqa: ARG002
        """Append this step's metrics to the shared state."""
        self.state.record_step(
            step=metrics.step, loss=metrics.loss, learning_rate=metrics.learning_rate
        )

    def on_train_end(self, trainer: Trainer) -> None:  # noqa: ARG002
        """Mark the run finished, unless it was already stopped or errored."""
        if self.state.status == "running":
            self.state.status = "finished"


class DashboardSampleCallback(SampleGenerationCallback):
    """A :class:`SampleGenerationCallback` that also writes its output into the dashboard state.

    Overrides :meth:`generate` rather than duplicating :meth:`on_step_end`'s
    scheduling logic, so a sample is still computed exactly once per firing.
    """

    def __init__(self, state: TrainingRunState, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.state = state

    def generate(self, trainer: Trainer) -> str:
        """Generate a sample and also stash it in the shared dashboard state."""
        text = super().generate(trainer)
        self.state.latest_sample = text
        return text
