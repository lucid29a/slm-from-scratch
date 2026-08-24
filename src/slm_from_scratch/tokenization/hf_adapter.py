"""An adapter wrapping HuggingFace's Rust-backed `tokenizers` library.

This is intentionally *not* used anywhere in the training or inference path -- the
whole point of this project is that the tokenizer is implemented by hand. It exists
solely as a fast, battle-tested oracle in the test suite: something to cross-check
:class:`~slm_from_scratch.tokenization.byte_level_bpe.ByteLevelBPETokenizer` against
on properties like "does encode/decode round-trip every input" and "is the learned
vocabulary size in the right ballpark", without having to trust our own trainer to
grade its own homework.

Requires the optional ``data`` extra (``pip install -e ".[data]"``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slm_from_scratch.core.exceptions import TokenizerError

__all__ = ["HFTokenizerAdapter"]


class HFTokenizerAdapter:
    """Thin wrapper around a byte-level BPE `tokenizers.Tokenizer`, for testing only."""

    def __init__(self, vocab_size: int) -> None:
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as exc:
            raise TokenizerError(
                "HFTokenizerAdapter requires the 'tokenizers' package; "
                "install with `pip install -e '.[data]'`"
            ) from exc

        self._Tokenizer = Tokenizer
        self._pre_tokenizers = pre_tokenizers
        self._trainers = trainers
        self._models = models
        self._decoders = decoders
        self._vocab_size = vocab_size
        self._tokenizer: Any = None

    def train(self, texts: Iterable[str]) -> HFTokenizerAdapter:
        """Train a byte-level BPE tokenizer on ``texts`` using the reference library."""
        tokenizer = self._Tokenizer(self._models.BPE(unk_token="<unk>"))
        byte_level = self._pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.pre_tokenizer = byte_level
        tokenizer.decoder = self._decoders.ByteLevel()
        # tokenizers' BpeTrainer.__init__ has no type stubs, hence no-untyped-call.
        trainer = self._trainers.BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=self._vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            # Match our own tokenizer's guarantee that the full 256-byte alphabet
            # is always in-vocabulary, not just the bytes this corpus happens to
            # contain -- otherwise the two trainers aren't comparable.
            initial_alphabet=self._pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(list(texts), trainer=trainer)
        self._tokenizer = tokenizer
        return self

    @property
    def vocab_size(self) -> int:
        """Actual learned vocabulary size (may be less than requested)."""
        assert self._tokenizer is not None, "call train() first"
        return int(self._tokenizer.get_vocab_size())

    def encode(self, text: str) -> list[int]:
        """Encode text with the reference tokenizer."""
        assert self._tokenizer is not None, "call train() first"
        return list(self._tokenizer.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        """Decode ids with the reference tokenizer."""
        assert self._tokenizer is not None, "call train() first"
        return str(self._tokenizer.decode(ids))
