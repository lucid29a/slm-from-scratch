"""Cross-checks our byte-level BPE against the reference HuggingFace `tokenizers`
implementation, used only as a test oracle (see hf_adapter.py docstring)."""

from __future__ import annotations

import pytest

from slm_from_scratch.tokenization.byte_level_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerConfig,
)

tokenizers = pytest.importorskip("tokenizers")

from slm_from_scratch.tokenization.hf_adapter import HFTokenizerAdapter  # noqa: E402


def test_reference_tokenizer_round_trips(toy_corpus: list[str]) -> None:
    ref = HFTokenizerAdapter(vocab_size=300).train(toy_corpus)
    text = "the quick brown fox"
    assert ref.decode(ref.encode(text)) == text


def test_our_vocab_size_is_in_the_same_ballpark_as_the_reference(toy_corpus: list[str]) -> None:
    ours = ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=300)).train(toy_corpus)
    ref = HFTokenizerAdapter(vocab_size=300).train(toy_corpus)

    # Both learn merges from the same corpus toward the same budget; on a corpus
    # this tiny they won't be bit-identical (tie-breaking differs, and ours stops
    # early once a pair's frequency drops below `min_pair_frequency` rather than
    # learning noise merges down to the last slot), but should land within a
    # small margin of each other -- not off by an order of magnitude, which
    # would indicate our trainer is fundamentally broken.
    assert abs(ours.vocab_size - ref.vocab_size) <= 20


def test_our_encoding_is_not_wildly_less_efficient_than_the_reference(
    toy_corpus: list[str],
) -> None:
    ours = ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=300)).train(toy_corpus)
    ref = HFTokenizerAdapter(vocab_size=300).train(toy_corpus)

    text = toy_corpus[0]
    ours_len = len(ours.encode(text))
    ref_len = len(ref.encode(text))
    assert ours_len <= ref_len * 1.5
