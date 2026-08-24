"""Learning-rate schedules: cosine decay with linear warmup, and warmup-stable-decay."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from slm_from_scratch.core.config import BaseConfig
from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.core.registry import Registry

__all__ = ["LR_SCHEDULES", "WSD", "CosineWithWarmup", "LRSchedule", "LRScheduleConfig"]


@dataclass(frozen=True, kw_only=True)
class LRScheduleConfig(BaseConfig):
    """Common configuration for a learning-rate schedule.

    Attributes:
        max_lr: Peak learning rate, reached at the end of warmup.
        min_lr: Floor the schedule decays to (never reaches exactly zero).
        warmup_steps: Number of steps to linearly ramp from 0 to ``max_lr``.
        total_steps: Total number of training steps the schedule spans.
    """

    max_lr: float
    min_lr: float = 0.0
    warmup_steps: int = 0
    total_steps: int = 1

    def validate(self) -> None:
        """Check the LR bounds and step counts are consistent."""
        if self.max_lr <= 0:
            raise ConfigError(f"max_lr must be positive, got {self.max_lr}")
        if self.min_lr < 0 or self.min_lr > self.max_lr:
            raise ConfigError(
                f"min_lr must be in [0, max_lr], got min_lr={self.min_lr}, max_lr={self.max_lr}"
            )
        if self.warmup_steps < 0:
            raise ConfigError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.total_steps <= 0:
            raise ConfigError(f"total_steps must be positive, got {self.total_steps}")
        if self.warmup_steps > self.total_steps:
            raise ConfigError(
                f"warmup_steps ({self.warmup_steps}) cannot exceed total_steps "
                f"({self.total_steps})"
            )


class LRSchedule(ABC):
    """Abstract base for a step -> learning-rate function."""

    def __init__(self, config: LRScheduleConfig) -> None:
        self.config = config

    def lr_at(self, step: int) -> float:
        """Return the learning rate for ``step`` (0-indexed).

        Handles warmup uniformly; delegates the post-warmup shape to
        :meth:`_decay_lr`.
        """
        config = self.config
        if step < config.warmup_steps:
            if config.warmup_steps == 0:
                return config.max_lr
            return config.max_lr * (step + 1) / config.warmup_steps
        return self._decay_lr(step)

    @abstractmethod
    def _decay_lr(self, step: int) -> float:
        """Return the learning rate for a ``step`` at or past the end of warmup."""
        raise NotImplementedError


LR_SCHEDULES: Registry[LRSchedule] = Registry("lr_schedule", LRSchedule)  # type: ignore[type-abstract]


@LR_SCHEDULES.register("cosine")
class CosineWithWarmup(LRSchedule):
    """Linear warmup, then cosine decay from ``max_lr`` down to ``min_lr``."""

    def _decay_lr(self, step: int) -> float:
        """Cosine-anneal from ``max_lr`` at the end of warmup to ``min_lr`` at ``total_steps``."""
        config = self.config
        if step >= config.total_steps:
            return config.min_lr
        progress = (step - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr + coeff * (config.max_lr - config.min_lr)


@LR_SCHEDULES.register("wsd")
class WSD(LRSchedule):
    """Warmup-Stable-Decay: constant peak LR, then a linear decay tail.

    Popularized as an alternative to cosine decay because the "stable" phase
    can be extended or cut short without needing to know the final step count
    in advance -- useful when a run's total token budget isn't fixed up front.
    The decay tail here is the final ``decay_fraction`` of ``total_steps``.
    """

    def __init__(self, config: LRScheduleConfig, *, decay_fraction: float = 0.1) -> None:
        super().__init__(config)
        if not 0.0 < decay_fraction <= 1.0:
            raise ConfigError(f"decay_fraction must be in (0, 1], got {decay_fraction}")
        self.decay_fraction = decay_fraction

    def _decay_lr(self, step: int) -> float:
        """Hold at ``max_lr`` until the decay tail, then decay linearly to ``min_lr``."""
        config = self.config
        decay_start = int(config.total_steps * (1.0 - self.decay_fraction))
        if step < decay_start:
            return config.max_lr
        if step >= config.total_steps:
            return config.min_lr
        progress = (step - decay_start) / max(config.total_steps - decay_start, 1)
        return config.max_lr - progress * (config.max_lr - config.min_lr)
