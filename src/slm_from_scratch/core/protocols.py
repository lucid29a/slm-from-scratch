"""Structural typing contracts enforced by ``mypy --strict`` across subsystems.

These :class:`typing.Protocol` classes describe the minimal shape a component must
have to plug into the training/eval/inference machinery, independent of which
abstract base class it happens to inherit from. Prefer the concrete ABCs
(``modeling.base.LanguageModel``, ``tokenization.base.Tokenizer``, ...) when writing
new components; these protocols exist so generic code (e.g. the CLI, the evaluator)
can type against behaviour rather than a specific class hierarchy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

__all__ = ["Loggable", "MetricLike", "SamplerLike", "TokenizerLike"]


@runtime_checkable
class TokenizerLike(Protocol):
    """Anything that can turn text into token ids and back."""

    @property
    def vocab_size(self) -> int:
        """Number of distinct token ids this tokenizer can produce."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token ids."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text."""
        ...


@runtime_checkable
class SamplerLike(Protocol):
    """Anything that turns next-token logits into a chosen token id."""

    def sample(
        self, logits: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Choose token ids from a ``(..., vocab_size)`` logits tensor."""
        ...


@runtime_checkable
class MetricLike(Protocol):
    """Anything that accumulates batches and reports a scalar result."""

    name: str

    def update(self, *, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate statistics from one batch."""
        ...

    def compute(self) -> float:
        """Return the metric's current value."""
        ...

    def reset(self) -> None:
        """Clear accumulated state."""
        ...


@runtime_checkable
class Loggable(Protocol):
    """Anything that can record scalar metrics and text samples during training."""

    def log_scalars(self, scalars: dict[str, float], *, step: int) -> None:
        """Record a batch of named scalar values at a training step."""
        ...

    def log_text(self, key: str, text: str, *, step: int) -> None:
        """Record a text sample (e.g. a generation) at a training step."""
        ...
