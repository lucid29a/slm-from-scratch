"""Tests for PrecisionPolicy."""

from __future__ import annotations

import pytest
import torch

from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.training.precision import PrecisionPolicy


def test_fp32_is_disabled_and_noop() -> None:
    policy = PrecisionPolicy("fp32", device_type="cpu")
    assert policy.enabled is False
    with policy.autocast():
        x = torch.randn(2, 2) @ torch.randn(2, 2)
    assert x.dtype == torch.float32


def test_bf16_is_enabled() -> None:
    policy = PrecisionPolicy("bf16", device_type="cpu")
    assert policy.enabled is True
    assert policy.dtype == torch.bfloat16


def test_rejects_unknown_precision() -> None:
    with pytest.raises(ConfigError, match="unknown precision"):
        PrecisionPolicy("int8", device_type="cpu")


def test_autocast_actually_casts_on_cpu() -> None:
    policy = PrecisionPolicy("bf16", device_type="cpu")
    with policy.autocast():
        out = torch.randn(2, 2) @ torch.randn(2, 2)
    assert out.dtype == torch.bfloat16
