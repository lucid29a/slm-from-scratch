"""Concrete text sources: local files and HuggingFace-hosted datasets.

:class:`TinyStoriesSource` and :class:`FineWebEduSource` are thin, opinionated
subclasses of :class:`HFDatasetSource` -- they exist so a config file can say
``type: tinystories`` instead of repeating the dataset name, split, and text
field every time. Both datasets are streamed rather than downloaded whole:
``datasets`` fetches shards lazily as you iterate, which is what makes a 3B-token
FineWeb-Edu sample tractable on a laptop with limited disk.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from slm_from_scratch.core.exceptions import DataError
from slm_from_scratch.data.base import SOURCES, TextSource, TextSourceConfig

__all__ = [
    "FineWebEduSource",
    "FineWebEduSourceConfig",
    "HFDatasetSource",
    "HFDatasetSourceConfig",
    "LocalFileSource",
    "LocalFileSourceConfig",
    "TinyStoriesSource",
    "TinyStoriesSourceConfig",
]


@dataclass(frozen=True, kw_only=True)
class LocalFileSourceConfig(TextSourceConfig):
    """Configuration for :class:`LocalFileSource`.

    Attributes:
        root: Directory to scan for text files.
        pattern: Glob pattern (relative to ``root``) selecting files.
        one_doc_per_line: If ``True``, each non-blank line of each matched file
            is its own document; if ``False``, each whole file is one document.
        encoding: Text encoding to read files with.
    """

    root: str
    pattern: str = "*.txt"
    one_doc_per_line: bool = False
    encoding: str = "utf-8"


@SOURCES.register("local_file")
class LocalFileSource(TextSource):
    """Reads documents from local ``.txt`` (or similarly plain-text) files."""

    config_cls: ClassVar[type[TextSourceConfig]] = LocalFileSourceConfig

    def _iter_documents(self) -> Iterator[str]:
        config = self.config
        assert isinstance(config, LocalFileSourceConfig)
        root = Path(config.root)
        if not root.is_dir():
            raise DataError(f"LocalFileSource root is not a directory: {root}")

        for path in sorted(root.rglob(config.pattern)):
            text = path.read_text(encoding=config.encoding)
            if config.one_doc_per_line:
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped:
                        yield stripped
            elif text.strip():
                yield text


@dataclass(frozen=True, kw_only=True)
class HFDatasetSourceConfig(TextSourceConfig):
    """Configuration for :class:`HFDatasetSource`.

    Attributes:
        dataset_name: HuggingFace Hub dataset identifier, e.g. ``"roneneldan/TinyStories"``.
        dataset_config: Optional dataset config/subset name.
        split: Dataset split to read.
        text_field: Column holding the document text.
        streaming: Stream shards lazily instead of downloading the whole dataset.
    """

    dataset_name: str
    dataset_config: str | None = None
    split: str = "train"
    text_field: str = "text"
    streaming: bool = True


@SOURCES.register("hf_dataset")
class HFDatasetSource(TextSource):
    """Streams documents from a HuggingFace Hub dataset.

    Requires the optional ``data`` extra (``pip install -e ".[data]"``).
    """

    config_cls: ClassVar[type[TextSourceConfig]] = HFDatasetSourceConfig

    def _iter_documents(self) -> Iterator[str]:
        config = self.config
        assert isinstance(config, HFDatasetSourceConfig)
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DataError(
                "HFDatasetSource requires the 'datasets' package; "
                "install with `pip install -e '.[data]'`"
            ) from exc

        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.split,
            streaming=config.streaming,
        )
        for example in dataset:
            text = example.get(config.text_field)
            if isinstance(text, str) and text.strip():
                yield text


@dataclass(frozen=True, kw_only=True)
class TinyStoriesSourceConfig(HFDatasetSourceConfig):
    """:class:`HFDatasetSourceConfig` defaulted to the TinyStories dataset."""

    dataset_name: str = "roneneldan/TinyStories"
    text_field: str = "text"


@SOURCES.register("tinystories")
class TinyStoriesSource(HFDatasetSource):
    """The TinyStories corpus: short, simple English stories for small models."""

    config_cls: ClassVar[type[TextSourceConfig]] = TinyStoriesSourceConfig


@dataclass(frozen=True, kw_only=True)
class FineWebEduSourceConfig(HFDatasetSourceConfig):
    """:class:`HFDatasetSourceConfig` defaulted to a FineWeb-Edu sample."""

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    text_field: str = "text"


@SOURCES.register("fineweb_edu")
class FineWebEduSource(HFDatasetSource):
    """A quality-filtered educational-web-text sample, used for the 150M run."""

    config_cls: ClassVar[type[TextSourceConfig]] = FineWebEduSourceConfig
