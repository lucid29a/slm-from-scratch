"""Tests for the attention strategies."""

from __future__ import annotations

import pytest
import torch

from slm_from_scratch.modeling.attention import (
    ATTENTION,
    CausalSelfAttention,
    GroupedQueryAttention,
    VanillaMultiHeadAttention,
)
from slm_from_scratch.modeling.positional import RotaryPositionalEmbedding

N_EMBD, N_HEAD, HEAD_DIM, BLOCK_SIZE = 32, 4, 8, 32


def make_positional() -> RotaryPositionalEmbedding:
    return RotaryPositionalEmbedding(
        n_embd=N_EMBD, n_head=N_HEAD, head_dim=HEAD_DIM, block_size=BLOCK_SIZE
    )


@pytest.mark.parametrize(
    ("cls", "n_kv_head"),
    [
        (VanillaMultiHeadAttention, N_HEAD),
        (CausalSelfAttention, N_HEAD),
        (GroupedQueryAttention, 2),
    ],
)
def test_output_shape_matches_input(cls: type, n_kv_head: int) -> None:
    attn = cls(n_embd=N_EMBD, n_head=N_HEAD, n_kv_head=n_kv_head, dropout=0.0, bias=False)
    x = torch.randn(2, 6, N_EMBD)
    out = attn(x, positional=make_positional())
    assert out.shape == x.shape


@pytest.mark.parametrize(
    ("cls", "n_kv_head"),
    [
        (VanillaMultiHeadAttention, N_HEAD),
        (CausalSelfAttention, N_HEAD),
        (GroupedQueryAttention, 2),
    ],
)
def test_gradients_flow(cls: type, n_kv_head: int) -> None:
    attn = cls(n_embd=N_EMBD, n_head=N_HEAD, n_kv_head=n_kv_head, dropout=0.0, bias=False)
    x = torch.randn(2, 6, N_EMBD, requires_grad=True)
    attn(x, positional=make_positional()).sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in attn.parameters())


@pytest.mark.parametrize(
    ("cls", "n_kv_head"),
    [
        (VanillaMultiHeadAttention, N_HEAD),
        (CausalSelfAttention, N_HEAD),
        (GroupedQueryAttention, 2),
    ],
)
def test_causal_invariance_output_does_not_see_future(cls: type, n_kv_head: int) -> None:
    torch.manual_seed(0)
    attn = cls(n_embd=N_EMBD, n_head=N_HEAD, n_kv_head=n_kv_head, dropout=0.0, bias=False)
    attn.eval()
    positional = make_positional()

    x = torch.randn(1, 8, N_EMBD)
    x_changed_tail = x.clone()
    x_changed_tail[:, 5:] = torch.randn(1, 3, N_EMBD)

    with torch.no_grad():
        out1 = attn(x, positional=positional)
        out2 = attn(x_changed_tail, positional=positional)

    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-6)
    assert not torch.allclose(out1[:, 5:], out2[:, 5:], atol=1e-6)


@pytest.mark.parametrize(
    ("cls", "n_kv_head"),
    [
        (VanillaMultiHeadAttention, N_HEAD),
        (CausalSelfAttention, N_HEAD),
        (GroupedQueryAttention, 2),
    ],
)
def test_kv_cache_matches_full_recompute(cls: type, n_kv_head: int) -> None:
    torch.manual_seed(1)
    attn = cls(n_embd=N_EMBD, n_head=N_HEAD, n_kv_head=n_kv_head, dropout=0.0, bias=False)
    attn.eval()
    positional = make_positional()

    x = torch.randn(1, 6, N_EMBD)

    with torch.no_grad():
        full_out = attn(x, positional=positional)

        from slm_from_scratch.modeling.kv_cache import LayerKVCache

        cache = LayerKVCache()
        outs = []
        for t in range(x.size(1)):
            outs.append(attn(x[:, t : t + 1], positional=positional, start_pos=t, kv_cache=cache))
        incremental_out = torch.cat(outs, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-4)


def test_gqa_degenerates_to_mha_when_n_kv_head_equals_n_head() -> None:
    torch.manual_seed(2)
    gqa = GroupedQueryAttention(
        n_embd=N_EMBD, n_head=N_HEAD, n_kv_head=N_HEAD, dropout=0.0, bias=False
    )
    assert gqa.n_kv_head == gqa.n_head
    x = torch.randn(1, 4, N_EMBD)
    out = gqa(x, positional=make_positional())
    assert out.shape == x.shape


def test_registry_contains_all_three() -> None:
    assert "vanilla_mha" in ATTENTION
    assert "causal_self_attention" in ATTENTION
    assert "grouped_query_attention" in ATTENTION
    assert "gqa" in ATTENTION  # alias
