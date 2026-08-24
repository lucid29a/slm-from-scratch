"""Tokenizers: char, word, hand-rolled BPE, and byte-level BPE (GPT-2 style).

Importing this package registers every concrete tokenizer and pre-tokenizer in the
``TOKENIZERS`` / ``PRETOKENIZERS`` registries, so ``TOKENIZERS.build("byte_level_bpe",
config)`` works as soon as ``slm_from_scratch.tokenization`` has been imported once.
"""

from __future__ import annotations

from slm_from_scratch.tokenization.base import (
    SPECIAL_TOKENS,
    TOKENIZERS,
    SpecialTokens,
    Tokenizer,
    TokenizerConfig,
)
from slm_from_scratch.tokenization.bpe import BPETokenizer, BPETokenizerConfig, BPETrainer
from slm_from_scratch.tokenization.byte_level_bpe import (
    ByteLevelBPETokenizer,
    ByteLevelBPETokenizerConfig,
)
from slm_from_scratch.tokenization.char import CharTokenizer, CharTokenizerConfig
from slm_from_scratch.tokenization.pretokenizer import (
    PRETOKENIZERS,
    ByteLevelPreTokenizer,
    PreTokenizer,
    WhitespacePreTokenizer,
)
from slm_from_scratch.tokenization.word import WordTokenizer, WordTokenizerConfig

__all__ = [
    "PRETOKENIZERS",
    "SPECIAL_TOKENS",
    "TOKENIZERS",
    "BPETokenizer",
    "BPETokenizerConfig",
    "BPETrainer",
    "ByteLevelBPETokenizer",
    "ByteLevelBPETokenizerConfig",
    "ByteLevelPreTokenizer",
    "CharTokenizer",
    "CharTokenizerConfig",
    "PreTokenizer",
    "SpecialTokens",
    "Tokenizer",
    "TokenizerConfig",
    "WhitespacePreTokenizer",
    "WordTokenizer",
    "WordTokenizerConfig",
]
