"""Tests for LayerNorm and RMSNorm."""

from __future__ import annotations

import pytest
import torch

from slm_from_scratch.modeling.normalization import NORMALIZATION, LayerNorm, RMSNorm


@pytest.mark.parametrize("cls", [LayerNorm, RMSNorm])
def test_output_shape_matches_input(cls: type) -> None:
    norm = cls(16)
    x = torch.randn(2, 5, 16)
    assert norm(x).shape == x.shape


def test_layernorm_output_has_zero_mean_unit_var() -> None:
    norm = LayerNorm(16)
    x = torch.randn(4, 16) * 5 + 3
    out = norm(x)
    assert out.mean(dim=-1).abs().max().item() < 1e-5
    assert (out.var(dim=-1, unbiased=False) - 1.0).abs().max().item() < 1e-4


def test_rmsnorm_does_not_center_mean() -> None:
    # RMSNorm rescales but does not subtract the mean, so a constant-shift input
    # should not come out zero-mean the way LayerNorm's would.
    norm = RMSNorm(16)
    x = torch.ones(1, 16) * 3.0
    out = norm(x)
    assert torch.allclose(out, norm.weight.unsqueeze(0), atol=1e-5)


def test_layernorm_bias_toggle() -> None:
    with_bias = LayerNorm(8, bias=True)
    without_bias = LayerNorm(8, bias=False)
    assert with_bias.bias is not None
    assert without_bias.bias is None


def test_registry_contains_both() -> None:
    assert "layernorm" in NORMALIZATION
    assert "rmsnorm" in NORMALIZATION


def test_gradients_flow() -> None:
    for cls in (LayerNorm, RMSNorm):
        norm = cls(8)
        x = torch.randn(2, 8, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None
