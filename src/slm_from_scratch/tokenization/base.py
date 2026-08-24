"""The tokenizer contract every concrete tokenizer implements.

A tokenizer turns text into a sequence of integer ids a model can embed, and back.
Everything downstream -- the data packer, the model's embedding table, the
generator's detokenization -- only ever talks to this interface, never to a
specific implementation, so :class:`CharTokenizer`, :class:`WordTokenizer`,
:class:`BPETokenizer`, and :class:`ByteLevelBPETokenizer` are interchangeable.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from slm_from_scratch.core.component import Component
from slm_from_scratch.core.config import BaseConfig
from slm_from_scratch.core.exceptions import TokenizerError
from slm_from_scratch.core.registry import Registry

__all__ = ["SPECIAL_TOKENS", "TOKENIZERS", "Tokenizer", "TokenizerConfig"]


@dataclass(frozen=True, kw_only=True)
class SpecialTokens:
    """The reserved tokens every tokenizer in this project carries.

    Reserved ids ``0..3`` are stable across tokenizer types so a checkpoint's
    special-token handling never depends on which tokenizer trained it.
    """

    unk: str = "<unk>"
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"

    def as_tuple(self) -> tuple[str, str, str, str]:
        """Return the four tokens in fixed id order ``(unk, pad, bos, eos)``."""
        return (self.unk, self.pad, self.bos, self.eos)


SPECIAL_TOKENS = SpecialTokens()


@dataclass(frozen=True, kw_only=True)
class TokenizerConfig(BaseConfig):
    """Common configuration shared by every tokenizer.

    Attributes:
        vocab_size: Target vocabulary size, special tokens included. Concrete
            tokenizers may end up with a slightly smaller vocabulary than
            requested (e.g. a byte-level BPE run out of merge opportunities on a
            tiny corpus); they never exceed it.
        lowercase: Whether to lowercase text before tokenizing.
    """

    vocab_size: int
    lowercase: bool = False

    def validate(self) -> None:
        """Check that vocab_size can hold the four reserved special tokens."""
        if self.vocab_size < len(SPECIAL_TOKENS.as_tuple()):
            from slm_from_scratch.core.exceptions import ConfigError

            raise ConfigError(
                f"vocab_size={self.vocab_size} is too small to hold the "
                f"{len(SPECIAL_TOKENS.as_tuple())} reserved special tokens"
            )


class Tokenizer(Component[TokenizerConfig], ABC):
    """Abstract base for everything that maps text <-> token ids.

    Subclasses must implement :meth:`encode`, :meth:`decode`, and the
    save/load pair (:meth:`_state_dict` / :meth:`_load_state_dict`) that
    persists whatever vocabulary or merge table they learned.
    """

    #: Short registry key, e.g. "char", "bpe" -- set by subclasses for error messages.
    kind: ClassVar[str] = "tokenizer"
    #: The concrete TokenizerConfig subclass this tokenizer is built from, used by
    #: :meth:`load` to reconstruct a config before reconstructing the tokenizer.
    config_cls: ClassVar[type[TokenizerConfig]] = TokenizerConfig

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)
        self._id_to_token: list[str] = []
        self._token_to_id: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        """Number of distinct token ids this tokenizer can produce."""
        return len(self._id_to_token)

    @property
    def unk_id(self) -> int:
        """Id of the unknown-token placeholder."""
        return self._token_to_id[SPECIAL_TOKENS.unk]

    @property
    def bos_id(self) -> int:
        """Id of the beginning-of-sequence token."""
        return self._token_to_id[SPECIAL_TOKENS.bos]

    @property
    def eos_id(self) -> int:
        """Id of the end-of-sequence token."""
        return self._token_to_id[SPECIAL_TOKENS.eos]

    @property
    def pad_id(self) -> int:
        """Id of the padding token."""
        return self._token_to_id[SPECIAL_TOKENS.pad]

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token ids (no BOS/EOS added).

        Args:
            text: The text to encode.

        Returns:
            A list of token ids.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text.

        Args:
            ids: Token ids previously produced by :meth:`encode` (or including
                special tokens).

        Returns:
            The decoded text.
        """
        raise NotImplementedError

    def encode_with_bos_eos(self, text: str) -> list[int]:
        """Encode ``text`` and wrap it with BOS/EOS ids.

        Args:
            text: The text to encode.

        Returns:
            ``[bos_id, *encode(text), eos_id]``.
        """
        return [self.bos_id, *self.encode(text), self.eos_id]

    def token_to_id(self, token: str) -> int:
        """Look up a single token's id, falling back to :attr:`unk_id`."""
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        """Look up a single id's token string.

        Raises:
            TokenizerError: If ``token_id`` is out of range.
        """
        if not 0 <= token_id < len(self._id_to_token):
            raise TokenizerError(
                f"token id {token_id} is out of range for a vocab of size "
                f"{len(self._id_to_token)}"
            )
        return self._id_to_token[token_id]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Persist this tokenizer's config and learned vocabulary to disk.

        Args:
            path: Destination JSON file; parent directories are created.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "kind": self.kind,
            "config": self.config.to_dict(),
            "state": self._state_dict(),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        """Load a tokenizer of any registered kind from a JSON file saved by :meth:`save`.

        Args:
            path: Path to the JSON file.

        Returns:
            A reconstructed tokenizer instance of the appropriate concrete type.

        Raises:
            TokenizerError: If the file's ``kind`` is not registered.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = payload["kind"]
        if kind not in TOKENIZERS:
            raise TokenizerError(f"cannot load tokenizer of unknown kind {kind!r}")
        tokenizer_cls = TOKENIZERS.get(kind)
        config = tokenizer_cls.config_cls.from_dict(payload["config"])
        instance = tokenizer_cls(config)
        instance._load_state_dict(payload["state"])
        return instance

    @abstractmethod
    def _state_dict(self) -> dict[str, Any]:
        """Return whatever this tokenizer needs to reconstruct itself (JSON-safe)."""
        raise NotImplementedError

    @abstractmethod
    def _load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore this tokenizer's learned state from :meth:`_state_dict` output."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers for subclasses building a vocabulary
    # ------------------------------------------------------------------ #
    def _install_vocab(self, tokens: list[str]) -> None:
        """Set the id<->token tables, asserting special tokens occupy ids 0-3.

        Args:
            tokens: The full ordered vocabulary; the first four entries must be
                the reserved special tokens.
        """
        expected = list(SPECIAL_TOKENS.as_tuple())
        if tokens[: len(expected)] != expected:
            raise TokenizerError(
                f"vocabulary must start with the special tokens {expected}, got "
                f"{tokens[: len(expected)]}"
            )
        self._id_to_token = list(tokens)
        self._token_to_id = {tok: i for i, tok in enumerate(tokens)}


# See the matching comment in pretokenizer.py: this is a registry-pattern false
# positive, not an actual attempt to instantiate the abstract Tokenizer class.
TOKENIZERS: Registry[Tokenizer] = Registry("tokenizer", Tokenizer)  # type: ignore[type-abstract]
