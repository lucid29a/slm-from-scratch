"""Shared fixtures for evaluation tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from slm_from_scratch.modeling import DecoderOnlyTransformer, ModelConfig


class SyntheticLMDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A tiny in-memory next-token dataset, for fast, network-free evaluation tests."""

    def __init__(self, n: int, block_size: int, vocab_size: int, seed: int = 0) -> None:
        self.n = n
        self.block_size = block_size
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n + block_size + 1,), generator=generator)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def make_tiny_model(seed: int = 42, **overrides: object) -> DecoderOnlyTransformer:
    torch.manual_seed(seed)
    defaults: dict[str, object] = {
        "vocab_size": 50,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 32,
        "block_size": 16,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return DecoderOnlyTransformer(ModelConfig(**defaults))  # type: ignore[arg-type]
