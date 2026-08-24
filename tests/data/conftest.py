"""Shared fixtures for data-pipeline tests."""

from __future__ import annotations

import pytest

from slm_from_scratch.tokenization.byte_level_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerConfig,
)


@pytest.fixture
def toy_corpus() -> list[str]:
    return [
        "The quick brown fox jumps over the lazy dog in the meadow.",
        "A curious cat watches birds from the windowsill every morning.",
        "Rain fell softly on the old wooden roof throughout the night.",
    ]


@pytest.fixture
def tokenizer(toy_corpus: list[str]) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=300)).train(toy_corpus)
