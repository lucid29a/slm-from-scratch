"""Character-level tokenizer: the simplest possible baseline.

One id per Unicode character seen during training, plus the four special tokens.
No merges, no sub-word structure -- included as the didactic starting point
(chapter 2 of the docs) and as a fast sanity-check tokenizer for tiny smoke tests.
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

__all__ = ["CharTokenizer", "CharTokenizerConfig"]


@dataclass(frozen=True, kw_only=True)
class CharTokenizerConfig(TokenizerConfig):
    """Configuration for :class:`CharTokenizer`."""


@TOKENIZERS.register("char")
class CharTokenizer(Tokenizer):
    """Maps each distinct character to its own token id.

    Build with :meth:`train` on a text corpus; unseen characters at inference
    time decode to ``<unk>``.
    """

    kind: ClassVar[str] = "char"
    config_cls: ClassVar[type[TokenizerConfig]] = CharTokenizerConfig

    def train(self, texts: Iterable[str]) -> CharTokenizer:
        """Learn the character vocabulary from a corpus.

        Args:
            texts: An iterable of training documents.

        Returns:
            ``self``, for chaining (``CharTokenizer(cfg).train(docs)``).
        """
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update((self.config.lowercase and text.lower()) or text)

        budget = self.config.vocab_size - len(SPECIAL_TOKENS.as_tuple())
        most_common = [ch for ch, _ in counts.most_common(budget)]
        vocab = [*SPECIAL_TOKENS.as_tuple(), *sorted(most_common)]
        self._install_vocab(vocab)
        return self

    def encode(self, text: str) -> list[int]:
        """Encode text one character at a time."""
        if self.config.lowercase:
            text = text.lower()
        return [self.token_to_id(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to characters, joined directly (no separators)."""
        return "".join(self.id_to_token(i) for i in ids)

    def _state_dict(self) -> dict[str, Any]:
        return {"vocab": self._id_to_token}

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self._install_vocab(list(state["vocab"]))
