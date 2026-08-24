"""Tests for GELUFeedForward and SwiGLUFeedForward."""

from __future__ import annotations

import pytest
import torch

from slm_from_scratch.modeling.feedforward import FEEDFORWARD, GELUFeedForward, SwiGLUFeedForward


@pytest.mark.parametrize("cls", [GELUFeedForward, SwiGLUFeedForward])
def test_output_shape_matches_input(cls: type) -> None:
    ffn = cls(16, hidden_multiplier=4.0, dropout=0.0, bias=False)
    x = torch.randn(2, 5, 16)
    assert ffn(x).shape == x.shape


@pytest.mark.parametrize("cls", [GELUFeedForward, SwiGLUFeedForward])
def test_gradients_flow(cls: type) -> None:
    ffn = cls(16, hidden_multiplier=4.0, dropout=0.0, bias=False)
    x = torch.randn(2, 16, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in ffn.parameters())


def test_registry_contains_both() -> None:
    assert "gelu" in FEEDFORWARD
    assert "swiglu" in FEEDFORWARD


def test_swiglu_has_three_weight_matrices_gelu_has_two() -> None:
    gelu = GELUFeedForward(16, hidden_multiplier=4.0, dropout=0.0, bias=False)
    swiglu = SwiGLUFeedForward(16, hidden_multiplier=4.0, dropout=0.0, bias=False)
    gelu_linears = [m for m in gelu.modules() if isinstance(m, torch.nn.Linear)]
    swiglu_linears = [m for m in swiglu.modules() if isinstance(m, torch.nn.Linear)]
    assert len(gelu_linears) == 2
    assert len(swiglu_linears) == 3
