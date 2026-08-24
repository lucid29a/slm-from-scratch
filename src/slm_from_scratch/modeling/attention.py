"""Attention strategies: explicit, fused-modern, and grouped-query.

All three share the same contract (:class:`AttentionBase`): given a residual
stream, a :class:`~slm_from_scratch.modeling.positional.PositionalEncodingBase`
to consult, and an optional :class:`~slm_from_scratch.modeling.kv_cache.LayerKVCache`,
produce the attended output. What differs between them is entirely internal:
whether Q/K/V are one fused projection or three separate ones, whether the
core computation is hand-written softmax-attention or
:func:`torch.nn.functional.scaled_dot_product_attention`, and how many
key/value heads there are relative to query heads.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from slm_from_scratch.core.registry import Registry
from slm_from_scratch.modeling.kv_cache import LayerKVCache
from slm_from_scratch.modeling.positional import PositionalEncodingBase

__all__ = [
    "ATTENTION",
    "AttentionBase",
    "CausalSelfAttention",
    "GroupedQueryAttention",
    "VanillaMultiHeadAttention",
]


class AttentionBase(nn.Module, ABC):
    """Abstract base for a transformer block's self-attention sub-layer.

    Args:
        n_embd: Model dimension.
        n_head: Number of query heads.
        n_kv_head: Number of key/value heads (equals ``n_head`` for plain MHA).
        dropout: Dropout probability on attention weights and the output projection.
        bias: Whether the linear projections carry a bias term.
    """

    def __init__(
        self, *, n_embd: int, n_head: int, n_kv_head: int, dropout: float, bias: bool
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.dropout_p = dropout
        self.bias = bias

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        positional: PositionalEncodingBase,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """Compute causal self-attention over ``x``.

        Args:
            x: ``(batch, seq_len, n_embd)`` input.
            positional: Positional encoding strategy; consulted for query/key
                rotation and/or an attention-score bias.
            start_pos: Absolute position of ``x``'s first token (nonzero during
                incremental decoding).
            kv_cache: If given, this layer's cache -- new keys/values are
                appended to it and the full history is attended over.

        Returns:
            ``(batch, seq_len, n_embd)`` attended output.
        """
        raise NotImplementedError


ATTENTION: Registry[AttentionBase] = Registry("attention", AttentionBase)  # type: ignore[type-abstract]


def _causal_mask(seq_len_q: int, seq_len_kv: int, *, device: torch.device) -> torch.Tensor:
    """Build an additive ``(1, 1, seq_len_q, seq_len_kv)`` causal mask.

    Query row ``i`` corresponds to absolute position ``(seq_len_kv - seq_len_q) + i``
    (so a single new query with a nonempty cache correctly attends to every
    cached position) and may attend to key columns at or before that position.
    """
    offset = seq_len_kv - seq_len_q
    q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1) + offset
    kv_pos = torch.arange(seq_len_kv, device=device).unsqueeze(0)
    allowed = kv_pos <= q_pos
    mask = torch.zeros(seq_len_q, seq_len_kv, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _fused_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Run scaled-dot-product attention with a causal mask and optional extra bias.

    Args:
        q: ``(batch, n_head, seq_len_q, head_dim)``.
        k: ``(batch, n_head, seq_len_kv, head_dim)`` (already head-count-matched to ``q``).
        v: Same shape as ``k``.
        bias: Optional ``(n_head, seq_len_q, seq_len_kv)`` additive bias (e.g. ALiBi).
        dropout_p: Attention-dropout probability.
        training: Whether dropout should actually be applied.

    Returns:
        ``(batch, n_head, seq_len_q, head_dim)`` attended output.
    """
    seq_len_q, seq_len_kv = q.size(2), k.size(2)
    effective_dropout = dropout_p if training else 0.0

    if bias is None and seq_len_q == seq_len_kv:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=effective_dropout)

    mask = _causal_mask(seq_len_q, seq_len_kv, device=q.device)
    if bias is not None:
        mask = mask + bias.unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=effective_dropout)


def _repeat_kv_heads(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """Repeat each of ``x``'s KV heads ``n_repeat`` times to match the query head count."""
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


@ATTENTION.register("vanilla_mha")
class VanillaMultiHeadAttention(AttentionBase):
    """The original (Vaswani et al. 2017) multi-head attention, spelled out explicitly.

    Separate Q/K/V projections and a hand-written softmax-attention computation
    -- no fused projection, no ``scaled_dot_product_attention`` kernel. Always
    plain multi-head attention (query and key/value head counts are equal);
    this is the S0 baseline in the paper's ablation ladder, deliberately kept
    as legible as possible rather than as fast as possible.
    """

    def __init__(
        self, *, n_embd: int, n_head: int, n_kv_head: int, dropout: float, bias: bool  # noqa: ARG002
    ) -> None:
        # Always plain MHA: n_kv_head is accepted for interface uniformity with
        # AttentionBase, but this strategy has no grouped-query variant.
        super().__init__(n_embd=n_embd, n_head=n_head, n_kv_head=n_head, dropout=dropout, bias=bias)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positional: PositionalEncodingBase,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """Explicit ``softmax(QK^T / sqrt(d) + mask) V`` attention."""
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x), batch, seq_len)
        k = self._split_heads(self.k_proj(x), batch, seq_len)
        v = self._split_heads(self.v_proj(x), batch, seq_len)

        q, k = positional.rotate_qk(q, k, start_pos=start_pos)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        seq_len_kv = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = _causal_mask(seq_len, seq_len_kv, device=x.device).squeeze(0).squeeze(0)
        scores = scores + mask
        bias = positional.attention_bias(seq_len, seq_len_kv, device=x.device)
        if bias is not None:
            scores = scores + bias.unsqueeze(0)

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        result: torch.Tensor = self.resid_dropout(self.out_proj(out))
        return result

    def _split_heads(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return x.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)


@ATTENTION.register("causal_self_attention")
class CausalSelfAttention(AttentionBase):
    """Fused-projection multi-head attention using PyTorch's SDPA kernel.

    A single ``Linear`` produces Q, K, and V together (one matmul instead of
    three), and the core attention computation is delegated to
    :func:`torch.nn.functional.scaled_dot_product_attention`, which dispatches
    to a fused, memory-efficient kernel. Functionally equivalent to
    :class:`VanillaMultiHeadAttention` with equal query/key head counts; this is
    the "modern but still plain-MHA" rung of the ablation ladder.
    """

    def __init__(
        self, *, n_embd: int, n_head: int, n_kv_head: int, dropout: float, bias: bool  # noqa: ARG002
    ) -> None:
        # Always plain MHA: n_kv_head is accepted for interface uniformity with
        # AttentionBase, but this strategy has no grouped-query variant.
        super().__init__(n_embd=n_embd, n_head=n_head, n_kv_head=n_head, dropout=dropout, bias=bias)
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positional: PositionalEncodingBase,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """Fused-QKV, SDPA-backed causal self-attention."""
        batch, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k = positional.rotate_qk(q, k, start_pos=start_pos)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        bias = positional.attention_bias(seq_len, k.size(2), device=x.device)
        out = _fused_attend(q, k, v, bias=bias, dropout_p=self.dropout_p, training=self.training)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        result: torch.Tensor = self.resid_dropout(self.out_proj(out))
        return result


@ATTENTION.register("grouped_query_attention", "gqa")
class GroupedQueryAttention(AttentionBase):
    """Grouped-Query Attention (Ainslie et al. 2023).

    Uses fewer key/value heads than query heads, with each KV head shared by
    ``n_head // n_kv_head`` query heads. This shrinks the KV cache -- the
    dominant memory cost of long-context inference -- proportionally, at a
    quality cost that in practice is close to free (Llama 2 70B, Mistral, and
    most current open models use it). With ``n_kv_head == n_head`` this
    degenerates exactly to plain multi-head attention.
    """

    def __init__(
        self, *, n_embd: int, n_head: int, n_kv_head: int, dropout: float, bias: bool
    ) -> None:
        super().__init__(
            n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, dropout=dropout, bias=bias
        )
        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positional: PositionalEncodingBase,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """SDPA-backed attention with ``n_kv_head`` key/value heads repeated to ``n_head``."""
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = positional.rotate_qk(q, k, start_pos=start_pos)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        n_repeat = self.n_head // self.n_kv_head
        k_expanded = _repeat_kv_heads(k, n_repeat)
        v_expanded = _repeat_kv_heads(v, n_repeat)

        bias = positional.attention_bias(seq_len, k.size(2), device=x.device)
        out = _fused_attend(
            q, k_expanded, v_expanded, bias=bias, dropout_p=self.dropout_p, training=self.training
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_head * self.head_dim)
        result: torch.Tensor = self.resid_dropout(self.out_proj(out))
        return result
