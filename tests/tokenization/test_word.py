"""Tests for WordTokenizer."""

from __future__ import annotations

from slm_from_scratch.tokenization.word import WordTokenizer, WordTokenizerConfig


def test_round_trip_known_text(toy_corpus: list[str]) -> None:
    tok = WordTokenizer(WordTokenizerConfig(vocab_size=64)).train(toy_corpus)
    text = "the quick fox"
    assert tok.decode(tok.encode(text)) == text


def test_oov_word_maps_to_unk(toy_corpus: list[str]) -> None:
    tok = WordTokenizer(WordTokenizerConfig(vocab_size=64)).train(toy_corpus)
    ids = tok.encode("elephant")
    assert tok.unk_id in ids


def test_vocab_size_respects_budget(toy_corpus: list[str]) -> None:
    tok = WordTokenizer(WordTokenizerConfig(vocab_size=8)).train(toy_corpus)
    assert tok.vocab_size <= 8


def test_lowercase_option(toy_corpus: list[str]) -> None:
    tok = WordTokenizer(WordTokenizerConfig(vocab_size=64, lowercase=True)).train(toy_corpus)
    assert tok.encode("THE") == tok.encode("the")
