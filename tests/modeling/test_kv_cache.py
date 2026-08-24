"""Tests for LayerKVCache and KVCache."""

from __future__ import annotations

import torch

from slm_from_scratch.modeling.kv_cache import KVCache, LayerKVCache


def test_layer_cache_starts_empty() -> None:
    cache = LayerKVCache()
    assert cache.seq_len == 0


def test_layer_cache_accumulates_across_updates() -> None:
    cache = LayerKVCache()
    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    full_k, _full_v = cache.update(k1, v1)
    assert full_k.shape == (1, 2, 3, 4)
    assert cache.seq_len == 3

    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    full_k, _full_v = cache.update(k2, v2)
    assert full_k.shape == (1, 2, 4, 4)
    assert cache.seq_len == 4
    assert torch.equal(full_k[:, :, :3], k1)
    assert torch.equal(full_k[:, :, 3:], k2)


def test_layer_cache_reset_clears_state() -> None:
    cache = LayerKVCache()
    cache.update(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
    cache.reset()
    assert cache.seq_len == 0


def test_kv_cache_indexes_layers_independently() -> None:
    cache = KVCache(n_layer=3)
    assert len(cache) == 3
    cache.layer(0).update(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
    assert cache.layer(0).seq_len == 2
    assert cache.layer(1).seq_len == 0


def test_kv_cache_seq_len_reflects_layer_zero() -> None:
    cache = KVCache(n_layer=2)
    cache.layer(0).update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4))
    assert cache.seq_len == 5


def test_kv_cache_reset_clears_every_layer() -> None:
    cache = KVCache(n_layer=2)
    for layer in range(2):
        cache.layer(layer).update(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
    cache.reset()
    assert cache.seq_len == 0
    assert cache.layer(1).seq_len == 0
