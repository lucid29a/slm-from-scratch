"""Tests for CosineWithWarmup and WSD."""

from __future__ import annotations

import math

import pytest

from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.training.lr_schedule import (
    LR_SCHEDULES,
    WSD,
    CosineWithWarmup,
    LRScheduleConfig,
)


def test_warmup_ramps_linearly_to_max_lr() -> None:
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1.0, warmup_steps=4, total_steps=100))
    assert schedule.lr_at(0) == pytest.approx(0.25)
    assert schedule.lr_at(1) == pytest.approx(0.5)
    assert schedule.lr_at(3) == pytest.approx(1.0)


def test_zero_warmup_steps_starts_at_max_lr() -> None:
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1.0, warmup_steps=0, total_steps=10))
    assert schedule.lr_at(0) == pytest.approx(1.0)


def test_cosine_decay_reaches_min_lr_at_total_steps() -> None:
    schedule = CosineWithWarmup(
        LRScheduleConfig(max_lr=1.0, min_lr=0.1, warmup_steps=0, total_steps=10)
    )
    assert schedule.lr_at(10) == pytest.approx(0.1)
    assert schedule.lr_at(1000) == pytest.approx(0.1)


def test_cosine_decay_midpoint() -> None:
    schedule = CosineWithWarmup(
        LRScheduleConfig(max_lr=1.0, min_lr=0.0, warmup_steps=0, total_steps=10)
    )
    # At the halfway point of a full cosine cycle, cos(pi/2) == 0 -> exactly the midpoint LR.
    mid = schedule.lr_at(5)
    assert mid == pytest.approx(0.5, abs=1e-2)


def test_cosine_is_monotonically_decreasing_after_warmup() -> None:
    schedule = CosineWithWarmup(LRScheduleConfig(max_lr=1.0, warmup_steps=0, total_steps=20))
    lrs = [schedule.lr_at(s) for s in range(21)]
    assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(len(lrs) - 1))


def test_wsd_holds_constant_during_stable_phase() -> None:
    schedule = WSD(
        LRScheduleConfig(max_lr=1.0, warmup_steps=0, total_steps=100), decay_fraction=0.2
    )
    assert schedule.lr_at(0) == pytest.approx(1.0)
    assert schedule.lr_at(50) == pytest.approx(1.0)
    assert schedule.lr_at(79) == pytest.approx(1.0)


def test_wsd_decays_linearly_in_tail() -> None:
    schedule = WSD(
        LRScheduleConfig(max_lr=1.0, min_lr=0.0, warmup_steps=0, total_steps=100),
        decay_fraction=0.2,
    )
    assert schedule.lr_at(100) == pytest.approx(0.0)
    # Halfway through the 20-step decay tail (steps 80-100): roughly half of max_lr.
    assert schedule.lr_at(90) == pytest.approx(0.5, abs=1e-6)


def test_wsd_rejects_bad_decay_fraction() -> None:
    with pytest.raises(ConfigError, match="decay_fraction"):
        WSD(LRScheduleConfig(max_lr=1.0, total_steps=10), decay_fraction=0.0)


def test_config_rejects_warmup_exceeding_total_steps() -> None:
    with pytest.raises(ConfigError, match="warmup_steps"):
        LRScheduleConfig(max_lr=1.0, warmup_steps=20, total_steps=10)


def test_config_rejects_min_lr_above_max_lr() -> None:
    with pytest.raises(ConfigError, match="min_lr"):
        LRScheduleConfig(max_lr=1.0, min_lr=2.0)


def test_registry_contains_both() -> None:
    assert "cosine" in LR_SCHEDULES
    assert "wsd" in LR_SCHEDULES


def test_cosine_matches_closed_form() -> None:
    config = LRScheduleConfig(max_lr=2.0, min_lr=0.2, warmup_steps=0, total_steps=8)
    schedule = CosineWithWarmup(config)
    step = 3
    expected = 0.2 + 0.5 * (1 + math.cos(math.pi * step / 8)) * (2.0 - 0.2)
    assert schedule.lr_at(step) == pytest.approx(expected)
