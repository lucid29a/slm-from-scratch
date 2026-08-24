"""Whitespace word-level tokenizer: the second step up from characters.

One id per distinct whitespace-delimited word, capped to the most frequent
``vocab_size`` words. Out-of-vocabulary words map to ``<unk>`` -- the exact
limitation that motivates sub-word tokenization (:mod:`bpe`) in the next chapter.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from slm_from_scratch.tokenization.base import (
    SPECIAL_TOKENS,
    TOKENIZERS,
    Tokenizer,
    TokenizerConfig,
)
from slm_from_scratch.tokenization.pretokenizer import WhitespacePreTokenizer

__all__ = ["WordTokenizer", "WordTokenizerConfig"]


@dataclass(frozen=True, kw_only=True)
class WordTokenizerConfig(TokenizerConfig):
    """Configuration for :class:`WordTokenizer`."""


@TOKENIZERS.register("word")
class WordTokenizer(Tokenizer):
    """Maps each distinct whitespace-delimited word to its own token id."""

    kind: ClassVar[str] = "word"
    config_cls: ClassVar[type[TokenizerConfig]] = WordTokenizerConfig

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)
        self._splitter = WhitespacePreTokenizer()

    def train(self, texts: Iterable[str]) -> WordTokenizer:
        """Learn the word vocabulary from a corpus, keeping the most frequent words.

        Args:
            texts: An iterable of training documents.

        Returns:
            ``self``, for chaining.
        """
        counts: Counter[str] = Counter()
        for text in texts:
            if self.config.lowercase:
                text = text.lower()
            counts.update(self._splitter.split(text))

        budget = self.config.vocab_size - len(SPECIAL_TOKENS.as_tuple())
        most_common = [word for word, _ in counts.most_common(budget)]
        vocab = [*SPECIAL_TOKENS.as_tuple(), *most_common]
        self._install_vocab(vocab)
        return self

    def encode(self, text: str) -> list[int]:
        """Encode text as a sequence of whole-word token ids."""
        if self.config.lowercase:
            text = text.lower()
        return [self.token_to_id(w) for w in self._splitter.split(text)]

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to words, joined with single spaces."""
        return self._splitter.join([self.id_to_token(i) for i in ids])

    def _state_dict(self) -> dict[str, Any]:
        return {"vocab": self._id_to_token}

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self._install_vocab(list(state["vocab"]))
