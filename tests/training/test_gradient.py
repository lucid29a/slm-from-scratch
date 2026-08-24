"""Tests for GradientAccumulator and GradientClipper."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.training.gradient import GradientAccumulator, GradientClipper


def test_accumulator_scales_loss() -> None:
    acc = GradientAccumulator(accumulation_steps=4)
    loss = torch.tensor(8.0)
    assert acc.scale_loss(loss).item() == pytest.approx(2.0)


def test_accumulator_signals_cycle_completion() -> None:
    acc = GradientAccumulator(accumulation_steps=3)
    assert acc.step() is False
    assert acc.step() is False
    assert acc.step() is True
    assert acc.step() is False  # new cycle


def test_accumulator_reset() -> None:
    acc = GradientAccumulator(accumulation_steps=2)
    acc.step()
    acc.reset()
    assert acc.step() is False
    assert acc.step() is True


def test_accumulator_rejects_nonpositive_steps() -> None:
    with pytest.raises(ConfigError, match="accumulation_steps"):
        GradientAccumulator(accumulation_steps=0)


def test_clipper_none_max_norm_is_noop() -> None:
    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    model(x).sum().backward()
    clipper = GradientClipper(max_norm=None)
    assert clipper.clip(model) is None


def test_clipper_reduces_large_gradients() -> None:
    model = nn.Linear(4, 4)
    with torch.no_grad():
        model.weight.grad = torch.full_like(model.weight, 100.0)
        model.bias.grad = torch.full_like(model.bias, 100.0)

    clipper = GradientClipper(max_norm=1.0)
    pre_clip_norm = clipper.clip(model)
    assert pre_clip_norm is not None
    assert pre_clip_norm > 1.0

    post_clip_norm = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters() if p.grad is not None]
    )
    assert float(post_clip_norm) == pytest.approx(1.0, abs=1e-4)


def test_clipper_rejects_nonpositive_max_norm() -> None:
    with pytest.raises(ConfigError, match="max_norm"):
        GradientClipper(max_norm=0.0)
