"""Tests for positional encoding strategies."""

from __future__ import annotations

import torch

from slm_from_scratch.modeling.positional import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)

N_EMBD, N_HEAD, HEAD_DIM, BLOCK_SIZE = 16, 4, 4, 32
_KW = {"n_embd": N_EMBD, "n_head": N_HEAD, "head_dim": HEAD_DIM, "block_size": BLOCK_SIZE}


def test_sinusoidal_encode_embeddings_shape() -> None:
    pe = SinusoidalPositionalEncoding(**_KW)
    x = torch.zeros(2, 10, N_EMBD)
    out = pe.encode_embeddings(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x)  # position info was actually added


def test_sinusoidal_is_deterministic_not_learned() -> None:
    pe1 = SinusoidalPositionalEncoding(**_KW)
    pe2 = SinusoidalPositionalEncoding(**_KW)
    assert torch.equal(pe1.table, pe2.table)


def test_learned_positional_embedding_is_trainable() -> None:
    pe = LearnedPositionalEmbedding(**_KW)
    assert any(p.requires_grad for p in pe.parameters())


def test_rope_preserves_shape() -> None:
    pe = RotaryPositionalEmbedding(**_KW)
    q = torch.randn(2, N_HEAD, 10, HEAD_DIM)
    k = torch.randn(2, N_HEAD, 10, HEAD_DIM)
    rq, rk = pe.rotate_qk(q, k)
    assert rq.shape == q.shape
    assert rk.shape == k.shape


def test_rope_preserves_vector_norm() -> None:
    # Rotation is norm-preserving: ||rotate(x)|| == ||x||.
    pe = RotaryPositionalEmbedding(**_KW)
    q = torch.randn(1, 1, 5, HEAD_DIM)
    k = torch.randn(1, 1, 5, HEAD_DIM)
    rq, _ = pe.rotate_qk(q, k)
    assert torch.allclose(q.norm(dim=-1), rq.norm(dim=-1), atol=1e-5)


def test_rope_dot_product_depends_only_on_relative_position() -> None:
    # The defining property of RoPE: <R(q, i), R(k, j)> depends only on (i - j).
    pe = RotaryPositionalEmbedding(**_KW)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    rq_a, rk_a = pe.rotate_qk(q, k, start_pos=3)  # relative offset (query at 3, key at 3)
    rq_b, rk_b = pe.rotate_qk(q, k, start_pos=10)  # same relative offset, shifted absolute position
    dot_a = (rq_a * rk_a).sum()
    dot_b = (rq_b * rk_b).sum()
    assert torch.allclose(dot_a, dot_b, atol=1e-4)


def test_rope_respects_start_pos_offset() -> None:
    pe = RotaryPositionalEmbedding(**_KW)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)
    rq0, _ = pe.rotate_qk(q, k, start_pos=0)
    rq5, _ = pe.rotate_qk(q, k, start_pos=5)
    assert not torch.allclose(rq0, rq5)


def test_alibi_bias_shape() -> None:
    pe = ALiBi(**_KW)
    bias = pe.attention_bias(5, 5, device=torch.device("cpu"))
    assert bias is not None
    assert bias.shape == (N_HEAD, 5, 5)


def test_alibi_bias_is_nonpositive_and_decreasing_with_distance() -> None:
    pe = ALiBi(**_KW)
    bias = pe.attention_bias(1, 5, device=torch.device("cpu"))
    assert bias is not None
    row = bias[0, 0]  # single query (last position) attending to 5 keys
    assert (row <= 0).all()
    # Bias should be monotonically non-increasing as key position moves away (more negative
    # for keys further in the past).
    assert torch.all(row[:-1] <= row[1:])


def test_alibi_slopes_geometric_for_power_of_two_heads() -> None:
    pe = ALiBi(**{**_KW, "n_head": 8})
    slopes = pe.slopes.tolist()
    ratios = [slopes[i] / slopes[i + 1] for i in range(len(slopes) - 1)]
    assert all(abs(r - ratios[0]) < 1e-4 for r in ratios)


def test_default_hooks_are_identity_for_non_applicable_strategies() -> None:
    pe = SinusoidalPositionalEncoding(**_KW)
    q = torch.randn(1, N_HEAD, 3, HEAD_DIM)
    k = torch.randn(1, N_HEAD, 3, HEAD_DIM)
    rq, rk = pe.rotate_qk(q, k)
    assert torch.equal(rq, q)
    assert torch.equal(rk, k)
    assert pe.attention_bias(3, 3, device=torch.device("cpu")) is None
