"""Tests for OptimizerFactory."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.training.optimizer import OptimizerConfig, OptimizerFactory


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=True)
        self.norm = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.norm(self.linear(x))
        return out


def test_splits_2d_and_1d_params_into_decay_groups() -> None:
    model = TinyModel()
    optimizer = OptimizerFactory(OptimizerConfig(weight_decay=0.1)).build(model)

    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    decay_shapes = {tuple(p.shape) for p in decay_group["params"]}
    no_decay_shapes = {tuple(p.shape) for p in no_decay_group["params"]}
    assert (4, 4) in decay_shapes  # linear.weight
    assert (4,) in no_decay_shapes  # linear.bias, norm.weight, norm.bias


def test_frozen_params_excluded() -> None:
    model = TinyModel()
    model.norm.weight.requires_grad_(False)
    optimizer = OptimizerFactory(OptimizerConfig()).build(model)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    assert not any(p is model.norm.weight for p in all_params)


def test_learning_rate_applied() -> None:
    model = TinyModel()
    optimizer = OptimizerFactory(OptimizerConfig(learning_rate=5e-4)).build(model)
    assert all(g["lr"] == 5e-4 for g in optimizer.param_groups)


def test_rejects_nonpositive_learning_rate() -> None:
    with pytest.raises(ConfigError, match="learning_rate"):
        OptimizerConfig(learning_rate=0.0)


def test_rejects_negative_weight_decay() -> None:
    with pytest.raises(ConfigError, match="weight_decay"):
        OptimizerConfig(weight_decay=-0.1)


def test_rejects_betas_out_of_range() -> None:
    with pytest.raises(ConfigError, match="beta"):
        OptimizerConfig(beta1=1.5)


def test_optimizer_step_updates_parameters() -> None:
    model = TinyModel()
    optimizer = OptimizerFactory(OptimizerConfig(learning_rate=1e-2, fused=False)).build(model)
    before = model.linear.weight.clone()

    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, model.linear.weight)
