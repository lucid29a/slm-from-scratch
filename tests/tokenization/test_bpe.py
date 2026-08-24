"""Tests for the hand-rolled BPE trainer and tokenizer."""

from __future__ import annotations

from slm_from_scratch.tokenization.base import Tokenizer
from slm_from_scratch.tokenization.bpe import BPETokenizer, BPETokenizerConfig, BPETrainer
from slm_from_scratch.tokenization.pretokenizer import WhitespacePreTokenizer


def test_trainer_learns_frequent_merge_first() -> None:
    # "ab" is by far the most frequent adjacent pair in this corpus.
    corpus = ["ab ab ab ab", "ab cd", "cd ab"]
    trainer = BPETrainer(WhitespacePreTokenizer(), min_pair_frequency=1)
    merges = trainer.train(corpus, num_merges=1)
    assert merges == [("a", "b")]


def test_trainer_respects_min_pair_frequency() -> None:
    corpus = ["a b"]  # single occurrence of the pair ("a", "b")
    trainer = BPETrainer(WhitespacePreTokenizer(), min_pair_frequency=5)
    merges = trainer.train(corpus, num_merges=10)
    assert merges == []


def test_trainer_stops_at_num_merges_cap() -> None:
    corpus = ["a b c d e f a b c d e f a b c d e f"]
    trainer = BPETrainer(WhitespacePreTokenizer(), min_pair_frequency=1)
    merges = trainer.train(corpus, num_merges=2)
    assert len(merges) == 2


def test_tokenizer_round_trip(toy_corpus: list[str]) -> None:
    tok = BPETokenizer(BPETokenizerConfig(vocab_size=100)).train(toy_corpus)
    for text in toy_corpus:
        assert tok.decode(tok.encode(text)) == text


def test_tokenizer_vocab_never_exceeds_budget(toy_corpus: list[str]) -> None:
    tok = BPETokenizer(BPETokenizerConfig(vocab_size=40)).train(toy_corpus)
    assert tok.vocab_size <= 40


def test_larger_vocab_yields_shorter_encodings(toy_corpus: list[str]) -> None:
    small = BPETokenizer(BPETokenizerConfig(vocab_size=40)).train(toy_corpus)
    large = BPETokenizer(BPETokenizerConfig(vocab_size=120)).train(toy_corpus)
    text = "the quick fox"
    assert len(large.encode(text)) <= len(small.encode(text))


def test_save_and_load_round_trip(toy_corpus: list[str], tmp_path) -> None:
    tok = BPETokenizer(BPETokenizerConfig(vocab_size=80)).train(toy_corpus)
    path = tmp_path / "bpe.json"
    tok.save(path)
    loaded = Tokenizer.load(path)

    assert isinstance(loaded, BPETokenizer)
    text = "the lazy fox"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
