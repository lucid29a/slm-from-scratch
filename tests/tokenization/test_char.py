"""Tests for CharTokenizer."""

from __future__ import annotations

from slm_from_scratch.tokenization.char import CharTokenizer, CharTokenizerConfig


def test_round_trip_known_text(toy_corpus: list[str]) -> None:
    tok = CharTokenizer(CharTokenizerConfig(vocab_size=64)).train(toy_corpus)
    text = "the quick fox"
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_occupy_first_four_ids(toy_corpus: list[str]) -> None:
    tok = CharTokenizer(CharTokenizerConfig(vocab_size=64)).train(toy_corpus)
    assert tok.id_to_token(0) == "<unk>"
    assert tok.id_to_token(1) == "<pad>"
    assert tok.id_to_token(2) == "<bos>"
    assert tok.id_to_token(3) == "<eos>"


def test_unseen_character_maps_to_unk(toy_corpus: list[str]) -> None:
    tok = CharTokenizer(CharTokenizerConfig(vocab_size=64)).train(toy_corpus)
    ids = tok.encode("\U0001f98a")  # an emoji never appears in the toy corpus
    assert ids == [tok.unk_id]


def test_vocab_size_respects_budget(toy_corpus: list[str]) -> None:
    tok = CharTokenizer(CharTokenizerConfig(vocab_size=10)).train(toy_corpus)
    assert tok.vocab_size <= 10


def test_encode_with_bos_eos(toy_corpus: list[str]) -> None:
    tok = CharTokenizer(CharTokenizerConfig(vocab_size=64)).train(toy_corpus)
    ids = tok.encode_with_bos_eos("dog")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_save_and_load_round_trip(toy_corpus: list[str], tmp_path) -> None:
    from slm_from_scratch.tokenization.base import Tokenizer

    tok = CharTokenizer(CharTokenizerConfig(vocab_size=64)).train(toy_corpus)
    path = tmp_path / "char_tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)

    assert isinstance(loaded, CharTokenizer)
    text = "the lazy dog"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
