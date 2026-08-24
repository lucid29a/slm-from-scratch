"""Tests for DecoderOnlyTransformer: the full assembled model."""

from __future__ import annotations

import pytest
import torch

from slm_from_scratch.modeling import MODELS, DecoderOnlyTransformer
from tests.modeling.conftest import make_config

ABLATION_LADDER_CONFIGS = [
    # S0: the 2017-original baseline.
    {
        "attention": "vanilla_mha",
        "positional_encoding": "sinusoidal",
        "normalization": "layernorm",
        "norm_placement": "post",
        "feedforward": "gelu",
        "weight_tying": False,
        "bias": True,
    },
    # S1: pre-norm.
    {
        "attention": "vanilla_mha",
        "positional_encoding": "sinusoidal",
        "normalization": "layernorm",
        "norm_placement": "pre",
        "feedforward": "gelu",
    },
    # S2: RMSNorm.
    {
        "attention": "vanilla_mha",
        "positional_encoding": "sinusoidal",
        "normalization": "rmsnorm",
        "norm_placement": "pre",
        "feedforward": "gelu",
    },
    # S3: RoPE.
    {
        "attention": "vanilla_mha",
        "positional_encoding": "rotary",
        "normalization": "rmsnorm",
        "norm_placement": "pre",
        "feedforward": "gelu",
    },
    # S4: SwiGLU.
    {
        "attention": "vanilla_mha",
        "positional_encoding": "rotary",
        "normalization": "rmsnorm",
        "norm_placement": "pre",
        "feedforward": "swiglu",
    },
    # S5: GQA.
    {
        "attention": "grouped_query_attention",
        "positional_encoding": "rotary",
        "normalization": "rmsnorm",
        "norm_placement": "pre",
        "feedforward": "swiglu",
        "n_kv_head": 2,
    },
    # S6: full modern stack (fused attention, weight tying, tuned init defaults).
    {
        "attention": "causal_self_attention",
        "positional_encoding": "rotary",
        "normalization": "rmsnorm",
        "norm_placement": "pre",
        "feedforward": "swiglu",
        "weight_tying": True,
    },
    # ALiBi, exercised separately from the ladder proper.
    {
        "attention": "causal_self_attention",
        "positional_encoding": "alibi",
        "normalization": "rmsnorm",
        "norm_placement": "pre",
        "feedforward": "swiglu",
    },
]


@pytest.mark.parametrize("overrides", ABLATION_LADDER_CONFIGS)
def test_every_ablation_ladder_combination_trains_one_step(overrides: dict[str, object]) -> None:
    config = make_config(**overrides)
    model = DecoderOnlyTransformer(config)
    x = torch.randint(0, config.vocab_size, (2, 10))
    y = torch.randint(0, config.vocab_size, (2, 10))

    logits, loss = model(x, y)
    assert logits.shape == (2, 10, config.vocab_size)
    assert loss is not None and loss.item() > 0

    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_forward_without_targets_returns_none_loss() -> None:
    config = make_config()
    model = DecoderOnlyTransformer(config)
    x = torch.randint(0, config.vocab_size, (1, 5))
    logits, loss = model(x)
    assert loss is None
    assert logits.shape == (1, 5, config.vocab_size)


def test_weight_tying_shares_the_same_tensor() -> None:
    config = make_config(weight_tying=True)
    model = DecoderOnlyTransformer(config)
    assert model.lm_head.weight is model.token_embedding.weight


def test_weight_tying_disabled_uses_separate_tensors() -> None:
    config = make_config(weight_tying=False)
    model = DecoderOnlyTransformer(config)
    assert model.lm_head.weight is not model.token_embedding.weight


def test_num_parameters_excludes_embedding_by_default() -> None:
    config = make_config(weight_tying=False)
    model = DecoderOnlyTransformer(config)
    total = model.num_parameters(non_embedding=False)
    non_emb = model.num_parameters(non_embedding=True)
    assert non_emb == total - config.vocab_size * config.n_embd


def test_num_parameters_with_tying_only_subtracts_once() -> None:
    config = make_config(weight_tying=True)
    model = DecoderOnlyTransformer(config)
    total = model.num_parameters(non_embedding=False)
    non_emb = model.num_parameters(non_embedding=True)
    # The tied weight is one Parameter object, counted once by .parameters();
    # subtracting its size once must not under- or over-count.
    assert non_emb == total - config.vocab_size * config.n_embd


def test_sequence_longer_than_block_size_without_cache_raises() -> None:
    config = make_config(block_size=8)
    model = DecoderOnlyTransformer(config)
    x = torch.randint(0, config.vocab_size, (1, 20))
    with pytest.raises(ValueError, match="block_size"):
        model(x)


def test_kv_cache_incremental_decoding_matches_full_recompute() -> None:
    torch.manual_seed(0)
    config = make_config(
        attention="grouped_query_attention",
        n_kv_head=2,
        positional_encoding="rotary",
        dropout=0.0,
    )
    model = DecoderOnlyTransformer(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        full_logits, _ = model(x)

        cache = model.new_kv_cache()
        outputs = []
        for t in range(x.size(1)):
            logits, _ = model(x[:, t : t + 1], kv_cache=cache)
            outputs.append(logits)
        incremental_logits = torch.cat(outputs, dim=1)

    assert torch.allclose(full_logits, incremental_logits, atol=1e-4)


def test_causal_mask_invariance_end_to_end() -> None:
    torch.manual_seed(3)
    config = make_config(dropout=0.0)
    model = DecoderOnlyTransformer(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (1, 10))
    x_tail_changed = x.clone()
    x_tail_changed[0, 7:] = torch.randint(0, config.vocab_size, (3,))

    with torch.no_grad():
        logits1, _ = model(x)
        logits2, _ = model(x_tail_changed)

    assert torch.equal(logits1[:, :7], logits2[:, :7])
    assert not torch.equal(logits1[:, 7:], logits2[:, 7:])


def test_model_is_registered() -> None:
    assert "decoder_only_transformer" in MODELS
