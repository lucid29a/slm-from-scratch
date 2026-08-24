"""Text-cleaning steps and the pipeline that chains them.

Each :class:`ProcessingStep` is a pure function of one document -> zero-or-one
documents, wrapped as an object so it can carry its own config and be composed.
:class:`ProcessingPipeline` runs a list of them in order over a stream of
documents; a document dropped by any step (returns ``None``) never reaches the
next one.
"""

from __future__ import annotations

import hashlib
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, get_args

from slm_from_scratch.core.component import Component
from slm_from_scratch.core.config import BaseConfig
from slm_from_scratch.core.registry import Registry

__all__ = [
    "MinHashDeduplicator",
    "MinHashDeduplicatorConfig",
    "PROCESSING_STEPS",
    "ProcessingPipeline",
    "ProcessingStep",
    "ProcessingStepConfig",
    "QualityFilter",
    "QualityFilterConfig",
    "UnicodeNormalizer",
    "UnicodeNormalizerConfig",
]


@dataclass(frozen=True, kw_only=True)
class ProcessingStepConfig(BaseConfig):
    """Marker base for a :class:`ProcessingStep`'s configuration."""


class ProcessingStep(Component[ProcessingStepConfig], ABC):
    """One document-cleaning stage: transform, or drop, a single document."""

    @abstractmethod
    def process(self, text: str) -> str | None:
        """Transform ``text``, or return ``None`` to drop it from the corpus."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any state accumulated across documents (stateless steps: no-op)."""


PROCESSING_STEPS: Registry[ProcessingStep] = Registry(
    "processing_step", ProcessingStep  # type: ignore[type-abstract]
)


# --------------------------------------------------------------------------- #
# Unicode normalization
# --------------------------------------------------------------------------- #
NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


@dataclass(frozen=True, kw_only=True)
class UnicodeNormalizerConfig(ProcessingStepConfig):
    """Configuration for :class:`UnicodeNormalizer`.

    Attributes:
        form: One of the standard Unicode normalization forms.
        strip_control_chars: Remove non-whitespace control characters (category
            ``Cc``), which are near-always corpus noise rather than content.
    """

    form: NormalizationForm = "NFC"
    strip_control_chars: bool = True

    def validate(self) -> None:
        """Check ``form`` is a normalization form :func:`unicodedata.normalize` accepts."""
        if self.form not in get_args(NormalizationForm):
            from slm_from_scratch.core.exceptions import ConfigError

            raise ConfigError(f"unknown Unicode normalization form: {self.form!r}")


@PROCESSING_STEPS.register("unicode_normalize")
class UnicodeNormalizer(ProcessingStep):
    """Normalizes Unicode representation and strips stray control characters."""

    def process(self, text: str) -> str | None:
        """Apply the configured normalization form and control-char stripping."""
        config = self.config
        assert isinstance(config, UnicodeNormalizerConfig)
        text = unicodedata.normalize(config.form, text)
        if config.strip_control_chars:
            text = "".join(
                ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc"
            )
        return text


# --------------------------------------------------------------------------- #
# Quality filtering
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, kw_only=True)
class QualityFilterConfig(ProcessingStepConfig):
    """Configuration for :class:`QualityFilter`.

    Attributes:
        min_chars: Drop documents shorter than this many characters.
        max_chars: Drop documents longer than this many characters (a single
            huge document can otherwise dominate a shard).
        min_alpha_ratio: Drop documents where fewer than this fraction of
            non-whitespace characters are alphabetic -- filters out data that's
            mostly markup, numbers, or symbol noise.
        max_line_repetition_ratio: Drop documents where the most common line
            makes up more than this fraction of all lines -- catches
            boilerplate-heavy scrapes (nav menus, repeated headers).
    """

    min_chars: int = 32
    max_chars: int = 100_000
    min_alpha_ratio: float = 0.5
    max_line_repetition_ratio: float = 0.5

    def validate(self) -> None:
        """Check the character bounds and ratios are internally consistent."""
        from slm_from_scratch.core.exceptions import ConfigError

        if self.min_chars < 0 or self.max_chars < self.min_chars:
            raise ConfigError(
                f"require 0 <= min_chars <= max_chars, got "
                f"min_chars={self.min_chars}, max_chars={self.max_chars}"
            )
        if not 0.0 <= self.min_alpha_ratio <= 1.0:
            raise ConfigError(f"min_alpha_ratio must be in [0, 1], got {self.min_alpha_ratio}")
        if not 0.0 <= self.max_line_repetition_ratio <= 1.0:
            raise ConfigError(
                f"max_line_repetition_ratio must be in [0, 1], got "
                f"{self.max_line_repetition_ratio}"
            )


@PROCESSING_STEPS.register("quality_filter")
class QualityFilter(ProcessingStep):
    """Drops documents that are too short, too long, too symbol-heavy, or too repetitive."""

    def process(self, text: str) -> str | None:
        """Return ``text`` unchanged, or ``None`` if any quality check fails."""
        config = self.config
        assert isinstance(config, QualityFilterConfig)

        length = len(text)
        if length < config.min_chars or length > config.max_chars:
            return None

        non_space = [ch for ch in text if not ch.isspace()]
        if non_space:
            alpha_ratio = sum(ch.isalpha() for ch in non_space) / len(non_space)
            if alpha_ratio < config.min_alpha_ratio:
                return None

        # A single-line document has nothing to compare against; only documents
        # with multiple lines can meaningfully be "mostly repeated boilerplate".
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            most_common_count = Counter(lines).most_common(1)[0][1]
            if most_common_count / len(lines) > config.max_line_repetition_ratio:
                return None

        return text


# --------------------------------------------------------------------------- #
# Approximate deduplication
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, kw_only=True)
class MinHashDeduplicatorConfig(ProcessingStepConfig):
    """Configuration for :class:`MinHashDeduplicator`.

    Attributes:
        num_hashes: Number of hash functions in each document's MinHash
            signature. More hashes -> a more accurate (and slower) similarity
            estimate.
        shingle_size: Size, in words, of the shingles hashed to build a
            signature.
        seed: Base seed for the hash-function family, for reproducibility.
    """

    num_hashes: int = 32
    shingle_size: int = 5
    seed: int = 0

    def validate(self) -> None:
        """Check the hash count and shingle size are positive."""
        from slm_from_scratch.core.exceptions import ConfigError

        if self.num_hashes <= 0:
            raise ConfigError(f"num_hashes must be positive, got {self.num_hashes}")
        if self.shingle_size <= 0:
            raise ConfigError(f"shingle_size must be positive, got {self.shingle_size}")


@PROCESSING_STEPS.register("minhash_dedup")
class MinHashDeduplicator(ProcessingStep):
    """Drops documents whose MinHash signature exactly matches one already seen.

    This is an *exact-signature* dedup (a document is dropped only if its full
    signature collides with a previous one), which in practice catches
    near-duplicates once the shingle size is small enough to be robust to minor
    edits, while remaining a single hash-table lookup per document -- no
    quadratic pairwise comparison. A production run with a very large corpus
    would bucket by signature bands (locality-sensitive hashing) for
    approximate matches; that refinement is a natural extension of this class,
    not a change to the pipeline around it.
    """

    def __init__(self, config: ProcessingStepConfig) -> None:
        super().__init__(config)
        self._seen_signatures: set[tuple[int, ...]] = set()

    def process(self, text: str) -> str | None:
        """Drop ``text`` if an identical MinHash signature was already seen."""
        signature = self._signature(text)
        if signature in self._seen_signatures:
            return None
        self._seen_signatures.add(signature)
        return text

    def reset(self) -> None:
        """Forget every signature seen so far."""
        self._seen_signatures.clear()

    def _signature(self, text: str) -> tuple[int, ...]:
        config = self.config
        assert isinstance(config, MinHashDeduplicatorConfig)
        words = text.split()
        shingles = {
            " ".join(words[i : i + config.shingle_size])
            for i in range(max(len(words) - config.shingle_size + 1, 1))
        }
        if not shingles:
            shingles = {text}

        return tuple(
            min(self._hash(shingle, seed) for shingle in shingles)
            for seed in range(config.seed, config.seed + config.num_hashes)
        )

    @staticmethod
    def _hash(shingle: str, seed: int) -> int:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, person=str(seed).encode())
        return int.from_bytes(digest.digest(), "big")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class ProcessingPipeline:
    """Runs an ordered list of :class:`ProcessingStep` instances over a document stream."""

    def __init__(self, steps: Iterable[ProcessingStep]) -> None:
        self._steps = list(steps)

    def run(self, documents: Iterable[str]) -> Iterator[str]:
        """Yield surviving documents, in order, after all steps have run.

        Args:
            documents: The raw input document stream.

        Yields:
            Documents that were not dropped by any step, in their final,
            transformed form.
        """
        for doc in documents:
            current: str | None = doc
            for step in self._steps:
                if current is None:
                    break
                current = step.process(current)
            if current is not None:
                yield current

    def reset(self) -> None:
        """Reset every step's accumulated state (e.g. the deduplicator's memory)."""
        for step in self._steps:
            step.reset()

    @property
    def steps(self) -> tuple[ProcessingStep, ...]:
        """The pipeline's steps, in execution order."""
        return tuple(self._steps)
