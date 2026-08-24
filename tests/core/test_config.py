"""Unit tests for slm_from_scratch.core.config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from slm_from_scratch.core.config import BaseConfig, load_yaml_with_extends
from slm_from_scratch.core.exceptions import ConfigError


@dataclass(frozen=True, kw_only=True)
class DummyConfig(BaseConfig):
    n_layer: int
    n_head: int
    n_embd: int = 64

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ConfigError("n_embd must be divisible by n_head")


def test_construct_valid_config() -> None:
    cfg = DummyConfig(n_layer=2, n_head=4, n_embd=64)
    assert cfg.n_layer == 2


def test_validate_runs_on_construction() -> None:
    with pytest.raises(ConfigError, match="divisible"):
        DummyConfig(n_layer=2, n_head=3, n_embd=64)


def test_frozen_cannot_mutate() -> None:
    cfg = DummyConfig(n_layer=2, n_head=4)
    with pytest.raises(AttributeError):
        cfg.n_layer = 99  # type: ignore[misc]


def test_from_dict_round_trip() -> None:
    cfg = DummyConfig.from_dict({"n_layer": 3, "n_head": 2, "n_embd": 32})
    assert cfg == DummyConfig(n_layer=3, n_head=2, n_embd=32)


def test_from_dict_rejects_unknown_field() -> None:
    with pytest.raises(ConfigError, match="unknown field"):
        DummyConfig.from_dict({"n_layer": 3, "n_head": 2, "bogus": 1})


def test_to_dict_returns_independent_copy() -> None:
    cfg = DummyConfig(n_layer=2, n_head=4)
    d = cfg.to_dict()
    d["n_layer"] = 999
    assert cfg.n_layer == 2


def test_replace_validates_the_new_instance() -> None:
    cfg = DummyConfig(n_layer=2, n_head=4, n_embd=64)
    cfg2 = cfg.replace(n_layer=5)
    assert cfg2.n_layer == 5
    with pytest.raises(ConfigError):
        cfg.replace(n_head=5)


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = DummyConfig(n_layer=6, n_head=8, n_embd=64)
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = DummyConfig.from_yaml(path)
    assert loaded == cfg


def test_from_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        DummyConfig.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_non_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml_with_extends(path)


def test_extends_merges_and_overrides(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text("n_layer: 2\nn_head: 4\nn_embd: 64\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\nn_layer: 8\n", encoding="utf-8")

    cfg = DummyConfig.from_yaml(child)
    assert cfg == DummyConfig(n_layer=8, n_head=4, n_embd=64)


def test_extends_chain_two_levels(tmp_path: Path) -> None:
    grandparent = tmp_path / "gp.yaml"
    grandparent.write_text("n_layer: 2\nn_head: 4\nn_embd: 32\n", encoding="utf-8")
    parent = tmp_path / "parent.yaml"
    parent.write_text("extends: gp.yaml\nn_embd: 64\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\nn_layer: 12\n", encoding="utf-8")

    cfg = DummyConfig.from_yaml(child)
    assert cfg == DummyConfig(n_layer=12, n_head=4, n_embd=64)


def test_extends_circular_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\nn_layer: 1\n", encoding="utf-8")
    b.write_text("extends: a.yaml\nn_head: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="circular"):
        load_yaml_with_extends(a)


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("n_layer: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml_with_extends(path)
