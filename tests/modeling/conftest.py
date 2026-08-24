"""Shared fixtures for modeling tests."""

from __future__ import annotations

import torch

from slm_from_scratch.modeling import ModelConfig


def make_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "vocab_size": 64,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
        "block_size": 16,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def seeded(seed: int = 0) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
