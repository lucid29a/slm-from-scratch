"""Gradient accumulation and clipping, as small, independently testable objects."""

from __future__ import annotations

import torch
from torch import nn

from slm_from_scratch.core.exceptions import ConfigError

__all__ = ["GradientAccumulator", "GradientClipper"]


class GradientAccumulator:
    """Splits one optimizer step across several micro-batches.

    Scales each micro-batch's loss down by the accumulation factor before
    ``backward()`` (so accumulated gradients average, rather than sum, across
    micro-batches) and tracks when a full accumulation cycle has completed.

    Args:
        accumulation_steps: Number of micro-batches per optimizer step.
    """

    def __init__(self, accumulation_steps: int) -> None:
        if accumulation_steps <= 0:
            raise ConfigError(
                f"accumulation_steps must be positive, got {accumulation_steps}"
            )
        self.accumulation_steps = accumulation_steps
        self._micro_step = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale ``loss`` so accumulated gradients average across micro-batches."""
        return loss / self.accumulation_steps

    def step(self) -> bool:
        """Record one micro-batch; return ``True`` if this completes an accumulation cycle."""
        self._micro_step += 1
        if self._micro_step >= self.accumulation_steps:
            self._micro_step = 0
            return True
        return False

    def reset(self) -> None:
        """Reset the micro-step counter (e.g. after a checkpoint resume)."""
        self._micro_step = 0


class GradientClipper:
    """Clips gradient norm to a maximum value; a no-op when ``max_norm`` is ``None``.

    Args:
        max_norm: Maximum gradient L2 norm; ``None`` disables clipping.
    """

    def __init__(self, max_norm: float | None) -> None:
        if max_norm is not None and max_norm <= 0:
            raise ConfigError(f"max_norm must be positive if set, got {max_norm}")
        self.max_norm = max_norm

    def clip(self, model: nn.Module) -> float | None:
        """Clip ``model``'s gradients in place; return the pre-clip total norm.

        Returns:
            The gradient norm before clipping, or ``None`` if clipping is disabled.
        """
        if self.max_norm is None:
            return None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        return float(norm)
