"""Normalization layers: hand-written LayerNorm and RMSNorm.

Both are registered under :data:`NORMALIZATION` so a config can pick either one
without touching :mod:`~slm_from_scratch.modeling.block`. This pair is also the
first rung of the paper's ablation ladder (S1 -> S2): swapping LayerNorm for
RMSNorm removes the mean-centering term and the bias, which is cheaper per step
and, empirically, rarely hurts quality at this scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from slm_from_scratch.core.registry import Registry

__all__ = ["NORMALIZATION", "LayerNorm", "NormalizationBase", "RMSNorm"]


class NormalizationBase(nn.Module, ABC):
    """Abstract base for a normalization layer over the last dimension.

    Args:
        dim: Size of the normalized (last) dimension.
        bias: Whether to include a learned additive bias term.
        eps: Numerical-stability epsilon.
    """

    def __init__(self, dim: int, *, bias: bool = False, eps: float = 1e-5) -> None:  # noqa: ARG002
        # `bias` is accepted here only so every subclass constructor shares one
        # signature; each subclass decides for itself whether to use it.
        super().__init__()
        self.dim = dim
        self.eps = eps

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension."""
        raise NotImplementedError


NORMALIZATION: Registry[NormalizationBase] = Registry(
    "normalization", NormalizationBase  # type: ignore[type-abstract]
)


@NORMALIZATION.register("layernorm")
class LayerNorm(NormalizationBase):
    """Layer normalization, implemented by hand (no ``torch.nn.functional.layer_norm``).

    Centers and rescales each sample to zero mean and unit variance over the
    last dimension, then applies a learned elementwise affine transform. This is
    the original (Ba et al. 2016) normalization used in the 2017 Transformer.
    """

    def __init__(self, dim: int, *, bias: bool = False, eps: float = 1e-5) -> None:
        super().__init__(dim, bias=bias, eps=eps)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, then apply the learned scale and (optional) shift."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = normalized * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


@NORMALIZATION.register("rmsnorm")
class RMSNorm(NormalizationBase):
    """Root Mean Square normalization (Zhang & Sennrich 2019).

    Rescales by the RMS of the last dimension without centering on the mean and
    without a bias term -- one fewer statistic to compute and one fewer
    parameter tensor than :class:`LayerNorm`, at effectively no quality cost in
    modern decoder-only transformers (Llama, Mistral, Qwen all use it).
    """

    def __init__(self, dim: int, *, bias: bool = False, eps: float = 1e-6) -> None:
        super().__init__(dim, bias=bias, eps=eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale by the root-mean-square over the last dimension, then apply the learned scale."""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
