"""The production tokenizer: byte-level BPE, GPT-2 style.

This is :class:`~slm_from_scratch.tokenization.bpe.BPETokenizer` specialised to the
byte-level pre-tokenizer, with one crucial addition: the full 256-byte alphabet is
always present in the vocabulary, even for byte values the training corpus never
happened to contain. That guarantee is what makes this tokenizer genuinely
"no-``<unk>``" -- any input, in any language, including binary-looking control
characters, encodes and decodes losslessly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from slm_from_scratch.tokenization.base import SPECIAL_TOKENS, TOKENIZERS
from slm_from_scratch.tokenization.bpe import BPETokenizer, BPETokenizerConfig, BPETrainer
from slm_from_scratch.tokenization.pretokenizer import ByteLevelPreTokenizer, byte_level_alphabet

__all__ = ["ByteLevelBPETokenizer", "ByteLevelBPETokenizerConfig"]


@dataclass(frozen=True, kw_only=True)
class ByteLevelBPETokenizerConfig(BPETokenizerConfig):
    """Configuration for :class:`ByteLevelBPETokenizer`. ``pretokenizer`` is fixed."""

    pretokenizer: str = "byte_level"

    def validate(self) -> None:
        """Also enforce that ``pretokenizer`` was not overridden to something else."""
        super().validate()
        if self.pretokenizer != "byte_level":
            from slm_from_scratch.core.exceptions import ConfigError

            raise ConfigError(
                "ByteLevelBPETokenizerConfig.pretokenizer must be 'byte_level'; "
                "use BPETokenizerConfig directly for other pretokenizers"
            )


@TOKENIZERS.register("byte_level_bpe")
class ByteLevelBPETokenizer(BPETokenizer):
    """GPT-2-style byte-level BPE: the tokenizer actually used to train the SLM.

    Guarantees the full 256-byte alphabet is in-vocabulary regardless of what the
    training corpus contains, so no input can ever produce an ``<unk>`` token.
    """

    kind: ClassVar[str] = "byte_level_bpe"
    config_cls: ClassVar[type[BPETokenizerConfig]] = ByteLevelBPETokenizerConfig

    def train(self, texts: Iterable[str]) -> ByteLevelBPETokenizer:
        """Learn merges from a corpus; the vocabulary always covers all 256 bytes.

        Args:
            texts: Training documents.

        Returns:
            ``self``, for chaining.
        """
        config = self.config
        assert isinstance(config, ByteLevelBPETokenizerConfig)
        texts = list(texts)
        if config.lowercase:
            texts = [t.lower() for t in texts]

        pretokenizer = self._pretokenizer
        assert isinstance(pretokenizer, ByteLevelPreTokenizer)

        # The full byte alphabet, not just the bytes observed in this corpus: a
        # byte value the corpus never used should still be representable.
        full_alphabet = sorted(byte_level_alphabet())

        specials = list(SPECIAL_TOKENS.as_tuple())
        budget = config.vocab_size - len(specials) - len(full_alphabet)
        trainer = BPETrainer(pretokenizer, min_pair_frequency=config.min_pair_frequency)
        merges = trainer.train(texts, num_merges=max(budget, 0))

        self._merges = merges
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}
        merged_symbols = [a + b for a, b in merges]
        self._install_vocab([*specials, *full_alphabet, *merged_symbols])
        return self
