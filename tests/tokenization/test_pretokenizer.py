"""Tests for the pre-tokenizer strategies."""

from __future__ import annotations

import pytest

from slm_from_scratch.tokenization.pretokenizer import (
    ByteLevelPreTokenizer,
    WhitespacePreTokenizer,
    byte_level_alphabet,
)


def test_whitespace_split_join_round_trip_single_spaces() -> None:
    pt = WhitespacePreTokenizer()
    text = "the quick brown fox"
    words = pt.split(text)
    assert pt.join(words) == text


def test_whitespace_to_symbols_and_back() -> None:
    pt = WhitespacePreTokenizer()
    for word in pt.split("the quick fox"):
        symbols = list(pt.to_symbols(word))
        assert pt.symbols_to_word(symbols) == word


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "",
        "leading and trailing spaces  ",
        "tabs\tand\nnewlines",
        "emoji \U0001f98a and 日本語",
        "".join(chr(c) for c in range(1, 256)),
    ],
)
def test_byte_level_round_trip_is_lossless_for_any_text(text: str) -> None:
    pt = ByteLevelPreTokenizer()
    chunks = pt.split(text)
    reconstructed = pt.join([pt.symbols_to_word(list(pt.to_symbols(c))) for c in chunks])
    assert reconstructed == text


def test_byte_level_alphabet_has_256_distinct_printable_chars() -> None:
    alphabet = byte_level_alphabet()
    assert len(alphabet) == 256
    assert len(set(alphabet)) == 256
    assert all(ch.isprintable() for ch in alphabet)
