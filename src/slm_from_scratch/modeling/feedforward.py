"""Feedforward (MLP) blocks: the original GELU MLP and the modern SwiGLU variant."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from slm_from_scratch.core.registry import Registry

__all__ = ["FEEDFORWARD", "FeedForwardBase", "GELUFeedForward", "SwiGLUFeedForward"]


class FeedForwardBase(nn.Module, ABC):
    """Abstract base for a transformer block's position-wise feedforward network.

    Args:
        n_embd: Model dimension (input and output size).
        hidden_multiplier: Hidden-layer size as a multiple of ``n_embd``.
        dropout: Dropout probability on the output projection.
        bias: Whether the linear layers carry a bias term.
    """

    def __init__(
        self, n_embd: int, *, hidden_multiplier: float, dropout: float, bias: bool
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.hidden_multiplier = hidden_multiplier
        self.dropout_p = dropout
        self.bias = bias

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feedforward transform to ``(..., n_embd)`` inputs."""
        raise NotImplementedError


FEEDFORWARD: Registry[FeedForwardBase] = Registry(
    "feedforward", FeedForwardBase  # type: ignore[type-abstract]
)


@FEEDFORWARD.register("gelu")
class GELUFeedForward(FeedForwardBase):
    """The original Transformer MLP: ``Linear -> GELU -> Linear``."""

    def __init__(
        self,
        n_embd: int,
        *,
        hidden_multiplier: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__(n_embd, hidden_multiplier=hidden_multiplier, dropout=dropout, bias=bias)
        hidden = int(n_embd * hidden_multiplier)
        self.fc_in = nn.Linear(n_embd, hidden, bias=bias)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``fc_out(dropout(gelu(fc_in(x))))``."""
        out: torch.Tensor = self.dropout(self.fc_out(self.activation(self.fc_in(x))))
        return out


@FEEDFORWARD.register("swiglu")
class SwiGLUFeedForward(FeedForwardBase):
    """SwiGLU (Shazeer 2020): a gated linear unit with a SiLU (Swish) gate.

    Splits into a value branch and a gate branch computed from the *same*
    input, multiplies them elementwise, then projects back down. To keep the
    parameter count comparable to a plain GELU MLP of the same
    ``hidden_multiplier`` (which has one up- and one down-projection, SwiGLU
    has two up-projections plus one down-projection), the hidden size is scaled
    by ``2/3`` -- the convention Llama and its successors use.
    """

    def __init__(
        self,
        n_embd: int,
        *,
        hidden_multiplier: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__(n_embd, hidden_multiplier=hidden_multiplier, dropout=dropout, bias=bias)
        hidden = int(n_embd * hidden_multiplier * 2 / 3)
        self.gate_proj = nn.Linear(n_embd, hidden, bias=bias)
        self.value_proj = nn.Linear(n_embd, hidden, bias=bias)
        self.activation = nn.SiLU()
        self.down_proj = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``down_proj(dropout(silu(gate_proj(x)) * value_proj(x)))``."""
        gated = self.activation(self.gate_proj(x)) * self.value_proj(x)
        out: torch.Tensor = self.dropout(self.down_proj(gated))
        return out
