"""Tests for ModelConfig validation."""

from __future__ import annotations

import pytest

from slm_from_scratch.core.exceptions import ConfigError
from tests.modeling.conftest import make_config


def test_valid_config_constructs() -> None:
    cfg = make_config()
    assert cfg.n_embd == 32


def test_rejects_n_embd_not_divisible_by_n_head() -> None:
    with pytest.raises(ConfigError, match="n_embd"):
        make_config(n_embd=33, n_head=4)


def test_rejects_n_head_not_multiple_of_n_kv_head() -> None:
    with pytest.raises(ConfigError, match="n_kv_head"):
        make_config(n_head=4, n_kv_head=3)


def test_accepts_valid_gqa_head_ratio() -> None:
    cfg = make_config(n_head=4, n_kv_head=2)
    assert cfg.effective_n_kv_head == 2


def test_none_n_kv_head_defaults_to_n_head() -> None:
    cfg = make_config(n_head=4, n_kv_head=None)
    assert cfg.effective_n_kv_head == 4


def test_rejects_nonpositive_n_layer() -> None:
    with pytest.raises(ConfigError, match="n_layer"):
        make_config(n_layer=0)


def test_rejects_nonpositive_block_size() -> None:
    with pytest.raises(ConfigError, match="block_size"):
        make_config(block_size=0)


def test_rejects_dropout_out_of_range() -> None:
    with pytest.raises(ConfigError, match="dropout"):
        make_config(dropout=1.0)


def test_rejects_bad_norm_placement() -> None:
    with pytest.raises(ConfigError, match="norm_placement"):
        make_config(norm_placement="sideways")


def test_head_dim_property() -> None:
    cfg = make_config(n_embd=32, n_head=4)
    assert cfg.head_dim == 8
