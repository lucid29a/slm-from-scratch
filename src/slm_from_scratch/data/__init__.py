"""The data pipeline: sources -> processing -> packed binary shards -> Dataset.

Importing this package registers every concrete :class:`TextSource` and
:class:`ProcessingStep` in the ``SOURCES`` / ``PROCESSING_STEPS`` registries.
"""

from __future__ import annotations

from slm_from_scratch.data.base import SOURCES, TextSource, TextSourceConfig
from slm_from_scratch.data.packing import (
    BinaryShardWriter,
    MemmapTokenDataset,
    SequencePacker,
    ShardManifest,
)
from slm_from_scratch.data.processing import (
    PROCESSING_STEPS,
    MinHashDeduplicator,
    MinHashDeduplicatorConfig,
    ProcessingPipeline,
    ProcessingStep,
    ProcessingStepConfig,
    QualityFilter,
    QualityFilterConfig,
    UnicodeNormalizer,
    UnicodeNormalizerConfig,
)
from slm_from_scratch.data.sources import (
    FineWebEduSource,
    FineWebEduSourceConfig,
    HFDatasetSource,
    HFDatasetSourceConfig,
    LocalFileSource,
    LocalFileSourceConfig,
    TinyStoriesSource,
    TinyStoriesSourceConfig,
)

__all__ = [
    "PROCESSING_STEPS",
    "SOURCES",
    "BinaryShardWriter",
    "FineWebEduSource",
    "FineWebEduSourceConfig",
    "HFDatasetSource",
    "HFDatasetSourceConfig",
    "LocalFileSource",
    "LocalFileSourceConfig",
    "MemmapTokenDataset",
    "MinHashDeduplicator",
    "MinHashDeduplicatorConfig",
    "ProcessingPipeline",
    "ProcessingStep",
    "ProcessingStepConfig",
    "QualityFilter",
    "QualityFilterConfig",
    "SequencePacker",
    "ShardManifest",
    "TextSource",
    "TextSourceConfig",
    "TinyStoriesSource",
    "TinyStoriesSourceConfig",
    "UnicodeNormalizer",
    "UnicodeNormalizerConfig",
]
