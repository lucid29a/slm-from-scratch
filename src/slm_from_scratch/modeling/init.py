"""Weight initialization strategies.

Kept as its own small Strategy hierarchy (rather than a method on the model)
because initialization is itself an ablatable, swappable design decision --
the paper's S6 rung is exactly "everything from S5, plus weight tying and a
tuned init."
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from torch import nn

from slm_from_scratch.core.registry import Registry
from slm_from_scratch.modeling.base import ModelConfig

__all__ = ["INITIALIZERS", "NormalWeightInit", "WeightInitializer"]


class WeightInitializer(ABC):
    """Abstract base for a model weight-initialization strategy."""

    @abstractmethod
    def initialize(self, model: nn.Module, config: ModelConfig) -> None:
        """Initialize ``model``'s parameters in place.

        Args:
            model: The freshly constructed model (uninitialized or
                framework-default-initialized).
            config: The model's configuration, e.g. for depth-dependent scaling.
        """
        raise NotImplementedError


INITIALIZERS: Registry[WeightInitializer] = Registry(
    "weight_initializer", WeightInitializer  # type: ignore[type-abstract]
)


@INITIALIZERS.register("normal")
class NormalWeightInit(WeightInitializer):
    """GPT-2-style init: small-std normal weights, scaled-down residual projections.

    Every ``Linear`` and ``Embedding`` weight is drawn from
    ``N(0, std^2)``; biases are zeroed. Linear layers that project *back into*
    the residual stream (``out_proj`` in attention, ``down_proj``/``fc_out`` in
    the feedforward) are additionally scaled by ``1 / sqrt(2 * n_layer)``, which
    keeps the residual stream's variance from growing with depth -- the same
    trick GPT-2 uses (Radford et al. 2019, section 2.3).

    Args:
        std: Base standard deviation for the normal initialization.
    """

    _RESIDUAL_PROJECTION_NAMES: frozenset[str] = frozenset({"out_proj", "down_proj", "fc_out"})

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def initialize(self, model: nn.Module, config: ModelConfig) -> None:
        """Apply the base normal init, then rescale residual output projections."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.std)

        residual_scale = self.std / math.sqrt(2 * config.n_layer)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in (
                self._RESIDUAL_PROJECTION_NAMES
            ):
                nn.init.normal_(module.weight, mean=0.0, std=residual_scale)
