"""Checkpointing: save and restore a whole training run, not just model weights.

A checkpoint that only saves ``model.state_dict()`` can resume *a* model, but
not *this run* -- the optimizer's Adam moments reset, the learning-rate
schedule restarts from step 0, and the data sampler picks a different sequence
of batches. :class:`CheckpointManager` saves model, optimizer, step count, and
RNG state together, so resuming from a checkpoint reproduces the loss curve an
uninterrupted run would have produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from slm_from_scratch.core.exceptions import CheckpointError

__all__ = ["CheckpointManager", "TrainingState"]

_CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt$")


@dataclass
class TrainingState:
    """Everything needed to resume training exactly where it left off.

    Attributes:
        step: The next step to run (i.e. one past the last completed step).
        model_state: ``model.state_dict()``.
        optimizer_state: ``optimizer.state_dict()``.
        rng_state: CPU RNG state (``torch.get_rng_state()``).
        cuda_rng_state: Per-device CUDA RNG states, if any GPU was in use.
        extra: Anything else the caller wants preserved (e.g. best-eval-loss-so-far).
    """

    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    rng_state: torch.Tensor
    cuda_rng_state: list[torch.Tensor] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Saves/loads :class:`TrainingState` snapshots as ``step_{N}.pt`` files.

    Args:
        checkpoint_dir: Directory checkpoints are written to and read from.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Snapshot training state at ``step`` and write it to disk.

        Args:
            step: The next step to run when this checkpoint is resumed from.
            model: The model whose ``state_dict()`` is saved.
            optimizer: The optimizer whose ``state_dict()`` is saved.
            extra: Additional caller-defined state to preserve.

        Returns:
            The path the checkpoint was written to.
        """
        state = TrainingState(
            step=step,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            rng_state=torch.get_rng_state(),
            cuda_rng_state=(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            extra=extra or {},
        )
        path = self.checkpoint_dir / f"step_{step:012d}.pt"
        torch.save(state, path)
        return path

    def load(self, path: str | Path) -> TrainingState:
        """Load a specific checkpoint file.

        Args:
            path: Path to a ``.pt`` file written by :meth:`save`.

        Returns:
            The loaded :class:`TrainingState`.

        Raises:
            CheckpointError: If the file is missing.
        """
        path = Path(path)
        if not path.is_file():
            raise CheckpointError(f"checkpoint not found: {path}")
        state: TrainingState = torch.load(path, weights_only=False)
        return state

    def load_latest(self) -> TrainingState | None:
        """Load the highest-step checkpoint in ``checkpoint_dir``, if any exists.

        Returns:
            The latest :class:`TrainingState`, or ``None`` if no checkpoint exists.
        """
        latest = self.latest_checkpoint_path()
        return None if latest is None else self.load(latest)

    def latest_checkpoint_path(self) -> Path | None:
        """Return the path of the highest-step checkpoint, or ``None`` if none exist."""
        candidates = [
            (int(match.group(1)), p)
            for p in self.checkpoint_dir.glob("step_*.pt")
            if (match := _CHECKPOINT_PATTERN.search(p.name)) is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0])[1]

    @staticmethod
    def restore(
        state: TrainingState, *, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        """Load a :class:`TrainingState` into a live model, optimizer, and the RNG.

        Args:
            state: A state previously produced by :meth:`save` or :meth:`load`.
            model: The model to load weights into (architecture must match).
            optimizer: The optimizer to load state into (must be built on the
                same parameters as when the checkpoint was saved).
        """
        model.load_state_dict(state.model_state)
        optimizer.load_state_dict(state.optimizer_state)
        torch.set_rng_state(state.rng_state)
        if state.cuda_rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state.cuda_rng_state)
