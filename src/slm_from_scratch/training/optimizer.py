"""Builds the AdamW optimizer with the standard decay/no-decay parameter-group split.

Weight decay should only pull down weight *matrices* -- not biases, and not
normalization scale parameters (LayerNorm/RMSNorm weights), which are 1-D and
whose whole job is to rescale, not to be regularized toward zero. Splitting
parameters into two groups by tensor dimensionality is the standard recipe
(used by GPT-2, Llama, and effectively every modern training script).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from slm_from_scratch.core.config import BaseConfig
from slm_from_scratch.core.exceptions import ConfigError

__all__ = ["OptimizerConfig", "OptimizerFactory"]


@dataclass(frozen=True, kw_only=True)
class OptimizerConfig(BaseConfig):
    """Configuration for the AdamW optimizer.

    Attributes:
        learning_rate: Peak learning rate (the LR schedule scales from this).
        weight_decay: Decay coefficient applied to 2-D-and-up parameters.
        beta1: AdamW's first moment decay.
        beta2: AdamW's second moment decay.
        eps: AdamW's numerical-stability epsilon.
        fused: Use PyTorch's fused AdamW kernel when running on CUDA.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    fused: bool = True

    def validate(self) -> None:
        """Check the learning rate and decay coefficients are sane."""
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not 0.0 < self.beta1 < 1.0 or not 0.0 < self.beta2 < 1.0:
            raise ConfigError(
                f"beta1/beta2 must be in (0, 1), got beta1={self.beta1}, beta2={self.beta2}"
            )


class OptimizerFactory:
    """Builds an AdamW optimizer from a model and an :class:`OptimizerConfig`."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def build(self, model: nn.Module) -> torch.optim.Optimizer:
        """Construct the optimizer, splitting parameters into decay/no-decay groups.

        Args:
            model: The model whose (trainable) parameters will be optimized.

        Returns:
            A configured :class:`torch.optim.AdamW` instance.
        """
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)

        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        use_fused = self.config.fused and torch.cuda.is_available()
        return torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            fused=use_fused,
        )
