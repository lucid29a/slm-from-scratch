"""Tests for SingleDeviceStrategy (DDPStrategy is untested -- see its docstring)."""

from __future__ import annotations

from torch import nn

from slm_from_scratch.training.distributed import SingleDeviceStrategy


def test_single_device_wrap_is_identity() -> None:
    strategy = SingleDeviceStrategy()
    model = nn.Linear(2, 2)
    assert strategy.wrap_model(model) is model
    assert strategy.unwrap_model(model) is model


def test_single_device_is_main_process() -> None:
    strategy = SingleDeviceStrategy()
    assert strategy.is_main_process is True


def test_single_device_world_size_is_one() -> None:
    strategy = SingleDeviceStrategy()
    assert strategy.world_size == 1


def test_single_device_barrier_is_noop() -> None:
    strategy = SingleDeviceStrategy()
    strategy.barrier()  # must not raise
