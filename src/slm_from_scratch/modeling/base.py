"""The model-wide configuration and the LanguageModel contract every architecture implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from slm_from_scratch.core.component import Component
from slm_from_scratch.core.config import BaseConfig
from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.core.registry import Registry

__all__ = ["MODELS", "LanguageModel", "ModelConfig"]


@dataclass(frozen=True, kw_only=True)
class ModelConfig(BaseConfig):
    """Everything needed to build a decoder-only transformer, choices included.

    The string fields (``attention``, ``positional_encoding``, ``normalization``,
    ``feedforward``) are registry keys, not code -- this is the config that makes
    the ablation study in the paper a matter of editing YAML, not editing Python.

    Attributes:
        vocab_size: Size of the token vocabulary (from the tokenizer).
        n_layer: Number of transformer blocks.
        n_head: Number of query attention heads.
        n_embd: Model (embedding/residual-stream) dimension.
        block_size: Maximum sequence length the model was trained for.
        n_kv_head: Number of key/value heads. ``None`` means "same as n_head"
            (plain multi-head attention); a value less than ``n_head`` enables
            grouped-query attention.
        dropout: Dropout probability applied in attention and feedforward blocks.
        attention: Registry key for the attention strategy.
        positional_encoding: Registry key for the positional encoding strategy.
        normalization: Registry key for the normalization strategy.
        norm_placement: ``"pre"`` (modern: norm before the sub-layer) or
            ``"post"`` (original 2017 Transformer: norm after the residual add).
        feedforward: Registry key for the feedforward strategy.
        ffn_hidden_multiplier: Feedforward hidden-dimension multiplier relative
            to ``n_embd``.
        weight_tying: Tie the input embedding and output projection weights.
        bias: Whether Linear/Norm layers carry a bias term (modern models often
            omit it; the original Transformer includes it).
    """

    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    n_kv_head: int | None = None
    dropout: float = 0.0
    attention: str = "causal_self_attention"
    positional_encoding: str = "rotary"
    normalization: str = "rmsnorm"
    norm_placement: str = "pre"
    feedforward: str = "swiglu"
    ffn_hidden_multiplier: float = 4.0
    weight_tying: bool = True
    bias: bool = False

    def validate(self) -> None:
        """Check dimensional and structural consistency of the architecture choices."""
        if self.n_embd % self.n_head != 0:
            raise ConfigError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_kv_head is not None and (
            self.n_kv_head <= 0 or self.n_head % self.n_kv_head != 0
        ):
            raise ConfigError(
                f"n_head ({self.n_head}) must be a positive multiple of "
                f"n_kv_head ({self.n_kv_head})"
            )
        if self.n_layer <= 0:
            raise ConfigError(f"n_layer must be positive, got {self.n_layer}")
        if self.block_size <= 0:
            raise ConfigError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.norm_placement not in {"pre", "post"}:
            raise ConfigError(
                f"norm_placement must be 'pre' or 'post', got {self.norm_placement!r}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.n_embd // self.n_head

    @property
    def effective_n_kv_head(self) -> int:
        """Number of KV heads, defaulting to ``n_head`` when unset (plain MHA)."""
        return self.n_kv_head if self.n_kv_head is not None else self.n_head


class LanguageModel(Component[ModelConfig], nn.Module, ABC):
    """Abstract base for anything that maps token ids to next-token logits."""

    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        Component.__init__(self, config)

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model.

        Args:
            input_ids: ``(batch, seq_len)`` token ids.
            targets: Optional ``(batch, seq_len)`` next-token ids; when given,
                the cross-entropy loss is computed and returned alongside the
                logits.

        Returns:
            A ``(logits, loss)`` pair; ``loss`` is ``None`` when ``targets`` is
            ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def num_parameters(self, *, non_embedding: bool = True) -> int:
        """Count trainable parameters.

        Args:
            non_embedding: Exclude the token-embedding table (and its tied
                output-projection twin, if weight tying is on) from the count --
                the conventional way model sizes are reported.

        Returns:
            The parameter count.
        """
        raise NotImplementedError


MODELS: Registry[LanguageModel] = Registry(
    "language_model", LanguageModel  # type: ignore[type-abstract]
)
