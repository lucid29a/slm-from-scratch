"""Positional encoding strategies.

Different encodings inject position information at different points in the
model, so this module's common interface is three hooks rather than one
``forward``:

* :meth:`~PositionalEncodingBase.encode_embeddings` -- add position information
  directly to the token embeddings, once, before the first block. Used by
  :class:`SinusoidalPositionalEncoding` and :class:`LearnedPositionalEmbedding`.
* :meth:`~PositionalEncodingBase.rotate_qk` -- rotate queries and keys inside
  every attention layer. Used by :class:`RotaryPositionalEmbedding` (RoPE).
* :meth:`~PositionalEncodingBase.attention_bias` -- an additive bias on raw
  attention scores, recomputed per layer. Used by :class:`ALiBi`.

Every strategy implements all three hooks; the two that don't apply to it are
no-ops. That lets :class:`~slm_from_scratch.modeling.attention.AttentionBase`
call all three unconditionally, regardless of which encoding is configured --
the Strategy pattern doing the work a chain of ``if encoding == "rope": ...``
branches would otherwise do.
"""

from __future__ import annotations

import math
from abc import ABC

import torch
from torch import nn

from slm_from_scratch.core.registry import Registry

__all__ = [
    "POSITIONAL_ENCODING",
    "ALiBi",
    "LearnedPositionalEmbedding",
    "PositionalEncodingBase",
    "RotaryPositionalEmbedding",
    "SinusoidalPositionalEncoding",
]


class PositionalEncodingBase(nn.Module, ABC):
    """Abstract base for a positional encoding strategy.

    Args:
        n_embd: Model dimension (for embedding-level encodings).
        n_head: Number of attention heads (for attention-bias encodings).
        head_dim: Per-head dimension (for RoPE's rotation).
        block_size: Maximum sequence length to precompute tables for.
    """

    def __init__(self, *, n_embd: int, n_head: int, head_dim: int, block_size: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = head_dim
        self.block_size = block_size

    def encode_embeddings(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:  # noqa: ARG002
        """Add position information to token embeddings. Default: identity (no-op)."""
        return x

    def rotate_qk(
        self, q: torch.Tensor, k: torch.Tensor, *, start_pos: int = 0  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries/keys by position. Default: identity (no-op)."""
        return q, k

    def attention_bias(
        self,
        seq_len_q: int,  # noqa: ARG002
        seq_len_kv: int,  # noqa: ARG002
        *,
        device: torch.device,  # noqa: ARG002
    ) -> torch.Tensor | None:
        """An additive ``(n_head, seq_len_q, seq_len_kv)`` bias on attention scores.

        Default: ``None`` (no bias). Every strategy implements this same signature;
        strategies that don't add a score bias (everything except ALiBi) simply
        ignore the arguments and return ``None``.
        """
        return None


POSITIONAL_ENCODING: Registry[PositionalEncodingBase] = Registry(
    "positional_encoding", PositionalEncodingBase
)


@POSITIONAL_ENCODING.register("sinusoidal")
class SinusoidalPositionalEncoding(PositionalEncodingBase):
    """The original Transformer's fixed sinusoidal position embeddings.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d))``, ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d))``,
    added directly to the token embeddings. Not learned, so it generalizes to
    sequence lengths beyond those seen in training in a way learned embeddings
    cannot -- one of the few advantages it retains over its modern successors.
    """

    # Declared here (not just via register_buffer) so mypy knows `self.table` is a
    # Tensor -- nn.Module's __getattr__ is typed as returning `Tensor | Module`.
    table: torch.Tensor

    def __init__(self, *, n_embd: int, n_head: int, head_dim: int, block_size: int) -> None:
        super().__init__(n_embd=n_embd, n_head=n_head, head_dim=head_dim, block_size=block_size)
        position = torch.arange(block_size).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        table = torch.zeros(block_size, n_embd)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("table", table, persistent=False)

    def encode_embeddings(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        """Add the precomputed sinusoidal table, sliced to the current positions."""
        seq_len = x.size(1)
        return x + self.table[start_pos : start_pos + seq_len].unsqueeze(0)


@POSITIONAL_ENCODING.register("learned")
class LearnedPositionalEmbedding(PositionalEncodingBase):
    """A learned embedding table, one vector per absolute position (GPT-2 style)."""

    def __init__(self, *, n_embd: int, n_head: int, head_dim: int, block_size: int) -> None:
        super().__init__(n_embd=n_embd, n_head=n_head, head_dim=head_dim, block_size=block_size)
        self.embedding = nn.Embedding(block_size, n_embd)

    def encode_embeddings(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        """Add the learned embedding for each position in the current window."""
        seq_len = x.size(1)
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        embedded: torch.Tensor = self.embedding(positions)
        return x + embedded.unsqueeze(0)


@POSITIONAL_ENCODING.register("rotary", "rope")
class RotaryPositionalEmbedding(PositionalEncodingBase):
    """Rotary Position Embedding (Su et al. 2021, "RoPE").

    Rather than adding a position vector, RoPE rotates each consecutive pair of
    query/key dimensions by an angle proportional to the token's position. The
    dot product of two rotated vectors then depends only on their *relative*
    position, which is what lets RoPE-based models extrapolate more gracefully
    to longer sequences than absolute encodings -- and why it's the encoding
    used by Llama, Mistral, Qwen, and effectively every modern open model.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, *, n_embd: int, n_head: int, head_dim: int, block_size: int) -> None:
        super().__init__(n_embd=n_embd, n_head=n_head, head_dim=head_dim, block_size=block_size)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(block_size).float()
        freqs = torch.outer(positions, inv_freq)  # (block_size, head_dim // 2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def rotate_qk(
        self, q: torch.Tensor, k: torch.Tensor, *, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the rotary transform to ``q`` and ``k``.

        Args:
            q: ``(batch, n_head, seq_len, head_dim)``.
            k: ``(batch, n_kv_head, seq_len, head_dim)``.
            start_pos: Absolute position of the first token in this window
                (nonzero during incremental decoding with a KV cache).

        Returns:
            The rotated ``(q, k)`` pair, same shapes as the input.
        """
        seq_len = q.size(2)
        cos = self.cos[start_pos : start_pos + seq_len]
        sin = self.sin[start_pos : start_pos + seq_len]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_head, seq_len, head_dim); cos/sin: (seq_len, head_dim // 2).
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        rotated1 = x1 * cos - x2 * sin
        rotated2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rotated1
        out[..., 1::2] = rotated2
        return out


@POSITIONAL_ENCODING.register("alibi")
class ALiBi(PositionalEncodingBase):
    """Attention with Linear Biases (Press et al. 2021).

    Adds a fixed, non-learned bias to attention scores that penalizes distant
    key positions linearly, with a per-head slope geometrically spaced across
    heads. No embedding-level or per-token rotation cost, and -- like sinusoidal
    encoding -- extrapolates to sequence lengths beyond those trained on.
    """

    slopes: torch.Tensor

    def __init__(self, *, n_embd: int, n_head: int, head_dim: int, block_size: int) -> None:
        super().__init__(n_embd=n_embd, n_head=n_head, head_dim=head_dim, block_size=block_size)
        slopes = torch.tensor(self._slopes(n_head), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)

    def attention_bias(
        self, seq_len_q: int, seq_len_kv: int, *, device: torch.device
    ) -> torch.Tensor:
        """Build the ``(n_head, seq_len_q, seq_len_kv)`` linear-distance bias."""
        # Query i (offset so the *last* query aligns with the *last* key) attends
        # to key j with bias -slope * (offset_i - j) for j <= offset_i.
        kv_pos = torch.arange(seq_len_kv, device=device)
        q_pos = torch.arange(seq_len_kv - seq_len_q, seq_len_kv, device=device)
        distance = (q_pos.unsqueeze(1) - kv_pos.unsqueeze(0)).clamp(min=0).float()
        slopes = self.slopes.to(device)
        return -slopes.view(-1, 1, 1) * distance.unsqueeze(0)

    @staticmethod
    def _slopes(n_head: int) -> list[float]:
        def power_of_2_slopes(n: int) -> list[float]:
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            return [start * (start**i) for i in range(n)]

        if math.log2(n_head).is_integer():
            return power_of_2_slopes(n_head)
        # Interpolate for head counts that aren't a power of 2 (ALiBi paper, appendix).
        closest_pow2 = 2 ** math.floor(math.log2(n_head))
        base = power_of_2_slopes(closest_pow2)
        extra = power_of_2_slopes(2 * closest_pow2)[0::2][: n_head - closest_pow2]
        return base + extra
