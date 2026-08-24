"""A single transformer block: attention + feedforward, each residual-wrapped.

Each sub-layer is wrapped in a residual connection and a normalization layer,
in either placement. ``norm_placement="pre"`` (the modern default) normalizes
*before* each
sub-layer, which empirically trains more stably at depth without a learning-rate
warmup crutch. ``norm_placement="post"`` normalizes *after* the residual add, as
in the original 2017 Transformer -- kept as the paper's S0/S1 ablation baseline.
"""

from __future__ import annotations

import torch
from torch import nn

from slm_from_scratch.modeling.attention import ATTENTION
from slm_from_scratch.modeling.base import ModelConfig
from slm_from_scratch.modeling.feedforward import FEEDFORWARD
from slm_from_scratch.modeling.kv_cache import LayerKVCache
from slm_from_scratch.modeling.normalization import NORMALIZATION
from slm_from_scratch.modeling.positional import PositionalEncodingBase

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """One decoder layer: self-attention and a feedforward network, each pre/post-normed.

    Args:
        config: The model configuration; determines which attention,
            normalization, and feedforward strategies this block uses.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_placement = config.norm_placement

        self.attn = ATTENTION.build(
            config.attention,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.effective_n_kv_head,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.ffn = FEEDFORWARD.build(
            config.feedforward,
            config.n_embd,
            hidden_multiplier=config.ffn_hidden_multiplier,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.norm_attn = NORMALIZATION.build(config.normalization, config.n_embd, bias=config.bias)
        self.norm_ffn = NORMALIZATION.build(config.normalization, config.n_embd, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positional: PositionalEncodingBase,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """Apply attention and feedforward sub-layers with residual connections.

        Args:
            x: ``(batch, seq_len, n_embd)`` input.
            positional: Positional encoding strategy, forwarded to attention.
            start_pos: Absolute position of ``x``'s first token.
            kv_cache: This layer's KV cache, if generating incrementally.

        Returns:
            ``(batch, seq_len, n_embd)`` output.
        """
        if self.norm_placement == "pre":
            x = x + self.attn(
                self.norm_attn(x), positional=positional, start_pos=start_pos, kv_cache=kv_cache
            )
            x = x + self.ffn(self.norm_ffn(x))
        else:
            attn_out = self.attn(x, positional=positional, start_pos=start_pos, kv_cache=kv_cache)
            x = self.norm_attn(x + attn_out)
            x = self.norm_ffn(x + self.ffn(x))
        return x
