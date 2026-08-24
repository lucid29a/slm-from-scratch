"""Tests for the production byte-level BPE tokenizer.

The central property under test is losslessness: any Unicode string, including
scripts and symbols absent from the training corpus, must encode and decode back
to itself exactly, with zero ``<unk>`` tokens.
"""

from __future__ import annotations

import pytest

from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.tokenization.base import Tokenizer
from slm_from_scratch.tokenization.byte_level_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerConfig,
)


@pytest.fixture
def trained_tokenizer(toy_corpus: list[str]) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=300)).train(toy_corpus)


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "",
        "   leading and trailing spaces   ",
        "line1\nline2\ttabbed",
        "MixedCASE with Numbers 12345 and punctuation!?.,;:",
        "unseen vocabulary entirely absent from training データ 日本語",
        "emoji stress test \U0001f98a\U0001f525\U00002764",
        "combining marks: café naïve",
    ],
)
def test_round_trip_is_lossless(trained_tokenizer: ByteLevelBPETokenizer, text: str) -> None:
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.decode(ids) == text


def test_no_unk_ever_appears(trained_tokenizer: ByteLevelBPETokenizer) -> None:
    text = "totally unfamiliar content: 空白文字 \U0001f98a ☃"
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.unk_id not in ids


def test_full_byte_alphabet_is_always_in_vocab(toy_corpus: list[str]) -> None:
    # A tiny vocab budget still must cover all 256 bytes + 4 specials, so training
    # should not be able to learn any merges at all.
    tok = ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=260)).train(toy_corpus)
    assert tok.vocab_size == 260  # 4 specials + 256 bytes, no room for merges


def test_config_rejects_non_byte_level_pretokenizer() -> None:
    with pytest.raises(ConfigError, match="byte_level"):
        ByteLevelBPETokenizerConfig(vocab_size=300, pretokenizer="whitespace")


def test_larger_vocab_compresses_repetitive_text(toy_corpus: list[str]) -> None:
    small = ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=270)).train(toy_corpus)
    large = ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=400)).train(toy_corpus)
    text = toy_corpus[0]
    assert len(large.encode(text)) <= len(small.encode(text))


def test_save_and_load_round_trip(trained_tokenizer: ByteLevelBPETokenizer, tmp_path) -> None:
    path = tmp_path / "byte_level_bpe.json"
    trained_tokenizer.save(path)
    loaded = Tokenizer.load(path)

    assert isinstance(loaded, ByteLevelBPETokenizer)
    text = "round trip through disk: café \U0001f98a"
    assert loaded.encode(text) == trained_tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
