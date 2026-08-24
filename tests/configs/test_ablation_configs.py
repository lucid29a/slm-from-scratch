"""Tests that the checked-in ablation ladder configs actually form a valid, correctly
incremented ladder -- each rung changes exactly the field the plan says it changes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from slm_from_scratch.modeling import ModelConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "ablation"

RUNG_FILES = [
    "s0_vanilla.yaml",
    "s1_prenorm.yaml",
    "s2_rmsnorm.yaml",
    "s3_rope.yaml",
    "s4_swiglu.yaml",
    "s5_gqa.yaml",
    "s6_full.yaml",
]


def _load_model_spec(filename: str) -> dict[str, object]:
    data = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    return dict(data["model"])


@pytest.mark.parametrize("filename", RUNG_FILES)
def test_each_rung_builds_a_valid_model_config(filename: str) -> None:
    spec = _load_model_spec(filename)
    config = ModelConfig.from_dict(spec)
    assert config.vocab_size == 4096


def test_all_rungs_share_the_same_dimensions_and_token_budget() -> None:
    specs = [_load_model_spec(f) for f in RUNG_FILES]
    for spec in specs:
        assert spec["vocab_size"] == 4096
        assert spec["n_layer"] == 16
        assert spec["n_head"] == 8
        assert spec["n_embd"] == 512
        assert spec["block_size"] == 256


def test_all_rungs_share_the_same_trainer_budget_and_seed() -> None:
    for filename in RUNG_FILES:
        data = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        assert data["trainer"]["max_steps"] == 6000
        assert data["trainer"]["micro_batch_size"] == 32
        assert data["trainer"]["seed"] == 1337
        assert data["data"]["shard_dir"] == "artifacts/cli_smoke/shards"


def test_s0_is_the_2017_vanilla_baseline() -> None:
    spec = _load_model_spec("s0_vanilla.yaml")
    assert spec["attention"] == "vanilla_mha"
    assert spec["positional_encoding"] == "learned"
    assert spec["normalization"] == "layernorm"
    assert spec["norm_placement"] == "post"
    assert spec["feedforward"] == "gelu"
    assert spec["weight_tying"] is False


def test_s1_changes_only_norm_placement_from_s0() -> None:
    s0, s1 = _load_model_spec("s0_vanilla.yaml"), _load_model_spec("s1_prenorm.yaml")
    diff = {k for k in s0 if s0[k] != s1.get(k)}
    assert diff == {"norm_placement"}
    assert s1["norm_placement"] == "pre"


def test_s2_changes_only_normalization_from_s1() -> None:
    s1, s2 = _load_model_spec("s1_prenorm.yaml"), _load_model_spec("s2_rmsnorm.yaml")
    diff = {k for k in s1 if s1[k] != s2.get(k)}
    assert diff == {"normalization"}
    assert s2["normalization"] == "rmsnorm"


def test_s3_changes_only_positional_encoding_from_s2() -> None:
    s2, s3 = _load_model_spec("s2_rmsnorm.yaml"), _load_model_spec("s3_rope.yaml")
    diff = {k for k in s2 if s2[k] != s3.get(k)}
    assert diff == {"positional_encoding"}
    assert s3["positional_encoding"] == "rotary"


def test_s4_changes_only_feedforward_from_s3() -> None:
    s3, s4 = _load_model_spec("s3_rope.yaml"), _load_model_spec("s4_swiglu.yaml")
    diff = {k for k in s3 if s3[k] != s4.get(k)}
    assert diff == {"feedforward"}
    assert s4["feedforward"] == "swiglu"


def test_s5_changes_only_attention_from_s4() -> None:
    s4, s5 = _load_model_spec("s4_swiglu.yaml"), _load_model_spec("s5_gqa.yaml")
    # n_kv_head is newly introduced here (absent/None in s4), which is part of
    # switching attention to a GQA-capable strategy, not a separate rung.
    diff = {k for k in s4 if s4.get(k) != s5.get(k)} | {
        k for k in s5 if k not in s4 and s5.get(k) is not None
    }
    assert diff == {"attention", "n_kv_head"}
    assert s5["attention"] == "grouped_query_attention"
    assert s5["n_kv_head"] == 4


def test_s6_changes_only_weight_tying_from_s5() -> None:
    s5, s6 = _load_model_spec("s5_gqa.yaml"), _load_model_spec("s6_full.yaml")
    diff = {k for k in s5 if s5[k] != s6.get(k)}
    assert diff == {"weight_tying"}
    assert s6["weight_tying"] is True


def test_s6_is_the_full_modern_stack() -> None:
    spec = _load_model_spec("s6_full.yaml")
    assert spec["attention"] == "grouped_query_attention"
    assert spec["positional_encoding"] == "rotary"
    assert spec["normalization"] == "rmsnorm"
    assert spec["norm_placement"] == "pre"
    assert spec["feedforward"] == "swiglu"
    assert spec["weight_tying"] is True
