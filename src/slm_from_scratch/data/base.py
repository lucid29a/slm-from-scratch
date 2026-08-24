"""The text-source contract: anything that can yield a stream of raw documents.

Every corpus this project trains on -- a folder of local text files, a
HuggingFace-hosted dataset streamed over the network, TinyStories, a FineWeb-Edu
sample -- implements this same tiny interface, so the rest of the data pipeline
(:mod:`slm_from_scratch.data.processing`, :mod:`slm_from_scratch.data.packing`)
never needs to know where its text came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from slm_from_scratch.core.component import Component
from slm_from_scratch.core.config import BaseConfig
from slm_from_scratch.core.registry import Registry

__all__ = ["SOURCES", "TextSource", "TextSourceConfig"]


@dataclass(frozen=True, kw_only=True)
class TextSourceConfig(BaseConfig):
    """Common configuration shared by every text source.

    Attributes:
        limit: If set, yield at most this many documents -- useful for smoke
            tests and for capping an otherwise-huge streamed dataset.
    """

    limit: int | None = None

    def validate(self) -> None:
        """Check that ``limit``, if set, is positive."""
        if self.limit is not None and self.limit <= 0:
            from slm_from_scratch.core.exceptions import ConfigError

            raise ConfigError(f"limit must be positive if set, got {self.limit}")


class TextSource(Component[TextSourceConfig], ABC):
    """Abstract base for anything that yields raw text documents."""

    #: The concrete TextSourceConfig subclass this source is built from -- lets
    #: generic code (e.g. the CLI) construct the right config type from a plain
    #: YAML mapping without a chain of isinstance checks.
    config_cls: ClassVar[type[TextSourceConfig]] = TextSourceConfig

    @abstractmethod
    def _iter_documents(self) -> Iterator[str]:
        """Yield raw documents, ignoring :attr:`TextSourceConfig.limit`.

        Subclasses implement this; :meth:`__iter__` applies the limit uniformly
        so every concrete source gets that behaviour for free.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        """Yield documents, honoring :attr:`TextSourceConfig.limit` if set."""
        limit = self.config.limit
        for i, doc in enumerate(self._iter_documents()):
            if limit is not None and i >= limit:
                return
            yield doc


SOURCES: Registry[TextSource] = Registry("text_source", TextSource)  # type: ignore[type-abstract]
