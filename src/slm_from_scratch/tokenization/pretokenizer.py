"""Pre-tokenization strategies: how raw text is split into "words" for BPE.

This is a small Strategy hierarchy on its own because it is the piece that decides
whether a tokenizer can represent *any* input losslessly. :class:`WhitespacePreTokenizer`
is simple and readable (good for the tutorial chapter on BPE) but cannot round-trip
whitespace runs or non-whitespace-separated scripts. :class:`ByteLevelPreTokenizer`
operates on the UTF-8 bytes of the text remapped to a printable alphabet -- the
GPT-2 recipe -- and is therefore reversible for *any* Unicode string, including
emoji, mixed scripts, and malformed-looking whitespace.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import regex

from slm_from_scratch.core.registry import Registry

__all__ = ["PRETOKENIZERS", "ByteLevelPreTokenizer", "PreTokenizer", "WhitespacePreTokenizer"]


class PreTokenizer(ABC):
    """Splits text into words and turns each word into an initial symbol sequence.

    A word's symbol sequence is what BPE merge rules are learned and applied over.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split ``text`` into an ordered list of word-level chunks.

        Args:
            text: Raw input text.

        Returns:
            Chunks whose concatenation, after :meth:`join`, reproduces ``text``.
        """
        raise NotImplementedError

    @abstractmethod
    def to_symbols(self, word: str) -> tuple[str, ...]:
        """Turn one word chunk into its initial (pre-merge) symbol sequence.

        Args:
            word: A single chunk returned by :meth:`split`.

        Returns:
            The symbols BPE merges will be learned/applied over.
        """
        raise NotImplementedError

    @abstractmethod
    def symbols_to_word(self, symbols: list[str]) -> str:
        """Invert :meth:`to_symbols`.

        Turns a (possibly merged) symbol sequence back into the word chunk it
        came from.
        """
        raise NotImplementedError

    def join(self, words: list[str]) -> str:
        """Concatenate decoded word chunks back into text. Default: plain concat."""
        return "".join(words)


class WhitespacePreTokenizer(PreTokenizer):
    """Splits on whitespace, operating on raw Unicode characters.

    Each chunk keeps its own single leading space (if any), the same trick
    :class:`ByteLevelPreTokenizer` uses, so plain concatenation reconstructs the
    text exactly for single-space-separated input. Runs of more than one
    whitespace character collapse to one space -- a known, documented limitation
    of this didactic tokenizer, not of :class:`ByteLevelPreTokenizer`.
    """

    _WORD_RE = re.compile(r" ?\S+")

    def split(self, text: str) -> list[str]:
        """Split into words, each carrying at most one leading space."""
        return self._WORD_RE.findall(text)

    def to_symbols(self, word: str) -> tuple[str, ...]:
        """Return the word as a tuple of individual characters."""
        return tuple(word)

    def symbols_to_word(self, symbols: list[str]) -> str:
        """Concatenate symbols (characters or merged multi-char pieces)."""
        return "".join(symbols)


def _bytes_to_unicode() -> dict[int, str]:
    """Build GPT-2's reversible byte<->printable-unicode-character mapping.

    Bytes that are already printable Latin-1 characters map to themselves; the
    remaining bytes (control characters, etc.) are remapped into the printable
    range starting at U+0100, so every one of the 256 byte values gets a distinct,
    printable, single-character representation. This is what lets a byte-level
    tokenizer represent literally any input file with zero ``<unk>`` tokens.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    byte_to_char = dict.fromkeys(printable)
    extra_index = 0
    mapping: dict[int, str] = {}
    for b in range(256):
        if b in byte_to_char:
            mapping[b] = chr(b)
        else:
            mapping[b] = chr(256 + extra_index)
            extra_index += 1
    return mapping


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def byte_level_alphabet() -> tuple[str, ...]:
    """Return the 256 printable byte-characters, one per possible byte value.

    Used to seed a byte-level BPE vocabulary with full byte coverage even for
    values absent from the training corpus.
    """
    return tuple(_BYTE_TO_UNICODE[b] for b in range(256))

# GPT-2's pre-tokenization regex: splits into contractions, runs of letters (each
# optionally preceded by one space), runs of digits, runs of "other" symbols, and
# runs of whitespace -- so merges never cross a word/punctuation/whitespace boundary.
# Uses the third-party `regex` module (not stdlib `re`) because \p{L}/\p{N} Unicode
# property escapes are not supported by `re`.
_GPT2_SPLIT_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class ByteLevelPreTokenizer(PreTokenizer):
    """GPT-2-style byte-level pre-tokenizer: reversible for any Unicode input.

    Text is UTF-8 encoded, each byte is remapped to a printable "byte character",
    and the result is split with a regex that keeps words, numbers, punctuation
    runs, and whitespace as separate chunks. Because every possible byte has a
    representation, encode -> decode is lossless and there is no ``<unk>`` for
    out-of-vocabulary *characters* (only for byte-symbol pairs the model was never
    trained to combine, which practice never actually hits since single bytes are
    always in vocab).
    """

    def split(self, text: str) -> list[str]:
        """Regex-split into GPT-2-style chunks, remapped to byte-characters.

        Each chunk's UTF-8 bytes are remapped to the printable byte-character
        alphabet.
        """
        chunks = _GPT2_SPLIT_PATTERN.findall(text)
        return ["".join(_BYTE_TO_UNICODE[b] for b in chunk.encode("utf-8")) for chunk in chunks]

    def to_symbols(self, word: str) -> tuple[str, ...]:
        """A byte-remapped chunk is already one symbol per byte-character."""
        return tuple(word)

    def symbols_to_word(self, symbols: list[str]) -> str:
        """Map byte-characters back to bytes and UTF-8 decode.

        Multi-character merged symbols are handled by iterating characters, since
        every merge is built out of single byte-characters transitively.
        """
        raw = bytes(_UNICODE_TO_BYTE[ch] for symbol in symbols for ch in symbol)
        return raw.decode("utf-8", errors="replace")

    def join(self, words: list[str]) -> str:
        """Chunks already carry their own leading-space marker; just concatenate."""
        return "".join(words)


# mypy flags an ABC passed as a registry's base_class as "type-abstract" because
# Registry.build() calls `cls(*args, **kwargs)` on a `type[T]`; the registry never
# actually instantiates the base class itself, only registered concrete subclasses.
PRETOKENIZERS: Registry[PreTokenizer] = Registry(
    "pretokenizer", PreTokenizer  # type: ignore[type-abstract]
)
PRETOKENIZERS.register_class("whitespace", WhitespacePreTokenizer)
PRETOKENIZERS.register_class("byte_level", ByteLevelPreTokenizer)
