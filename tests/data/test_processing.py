"""Tests for processing steps and the pipeline that chains them."""

from __future__ import annotations

import pytest

from slm_from_scratch.data.processing import (
    MinHashDeduplicator,
    MinHashDeduplicatorConfig,
    ProcessingPipeline,
    QualityFilter,
    QualityFilterConfig,
    UnicodeNormalizer,
    UnicodeNormalizerConfig,
)


# --------------------------------------------------------------------------- #
# UnicodeNormalizer
# --------------------------------------------------------------------------- #
def test_unicode_normalizer_composes_combining_characters() -> None:
    # "e" + combining acute accent (NFD) should normalize to precomposed "é" (NFC).
    decomposed = "é"
    step = UnicodeNormalizer(UnicodeNormalizerConfig(form="NFC"))
    assert step.process(decomposed) == "é"


def test_unicode_normalizer_strips_control_characters() -> None:
    step = UnicodeNormalizer(UnicodeNormalizerConfig())
    result = step.process("hello\x00\x07world\n\ttab kept")
    assert result == "helloworld\n\ttab kept"


def test_unicode_normalizer_keeps_newlines_and_tabs() -> None:
    step = UnicodeNormalizer(UnicodeNormalizerConfig())
    text = "line one\n\tindented"
    assert step.process(text) == text


# --------------------------------------------------------------------------- #
# QualityFilter
# --------------------------------------------------------------------------- #
def test_quality_filter_drops_too_short() -> None:
    step = QualityFilter(QualityFilterConfig(min_chars=50))
    assert step.process("too short") is None


def test_quality_filter_drops_too_long() -> None:
    step = QualityFilter(QualityFilterConfig(min_chars=0, max_chars=10))
    assert step.process("this text is definitely too long") is None


def test_quality_filter_drops_symbol_heavy_text() -> None:
    step = QualityFilter(QualityFilterConfig(min_chars=1, min_alpha_ratio=0.8))
    assert step.process("!@#$%^&*()_+-=1234567890") is None


def test_quality_filter_keeps_normal_prose() -> None:
    step = QualityFilter(QualityFilterConfig())
    text = "This is a perfectly normal sentence about everyday things."
    assert step.process(text) == text


def test_quality_filter_drops_repetitive_multiline_boilerplate() -> None:
    step = QualityFilter(QualityFilterConfig(min_chars=1, max_line_repetition_ratio=0.5))
    text = "\n".join(["Home | About | Contact"] * 10 + ["actual unique content here"])
    assert step.process(text) is None


def test_quality_filter_does_not_penalize_single_line_documents() -> None:
    # Regression test: a single-line document trivially has "100% repetition"
    # under a naive most-common-line/total-lines ratio; it must not be dropped
    # for that reason alone.
    step = QualityFilter(QualityFilterConfig())
    text = "A single sentence with no newlines at all in it whatsoever here."
    assert step.process(text) == text


def test_quality_filter_config_rejects_inverted_bounds() -> None:
    from slm_from_scratch.core.exceptions import ConfigError

    with pytest.raises(ConfigError, match="min_chars"):
        QualityFilterConfig(min_chars=100, max_chars=10)


# --------------------------------------------------------------------------- #
# MinHashDeduplicator
# --------------------------------------------------------------------------- #
def test_dedup_keeps_first_occurrence_drops_exact_repeat() -> None:
    step = MinHashDeduplicator(MinHashDeduplicatorConfig())
    text = "the quick brown fox jumps over the lazy dog again and again"
    assert step.process(text) == text
    assert step.process(text) is None


def test_dedup_keeps_distinct_documents() -> None:
    step = MinHashDeduplicator(MinHashDeduplicatorConfig())
    assert step.process("first distinct document about space travel") is not None
    assert step.process("second distinct document about deep sea life") is not None


def test_dedup_reset_forgets_state() -> None:
    step = MinHashDeduplicator(MinHashDeduplicatorConfig())
    text = "a document that will be seen twice, with a reset in between"
    assert step.process(text) == text
    step.reset()
    assert step.process(text) == text


# --------------------------------------------------------------------------- #
# ProcessingPipeline
# --------------------------------------------------------------------------- #
def test_pipeline_runs_steps_in_order_and_drops_via_any_step() -> None:
    pipeline = ProcessingPipeline(
        [
            UnicodeNormalizer(UnicodeNormalizerConfig()),
            QualityFilter(QualityFilterConfig(min_chars=20)),
            MinHashDeduplicator(MinHashDeduplicatorConfig()),
        ]
    )
    docs = [
        "This is a sufficiently long first document about gardening tips.",
        "too short",
        "This is a sufficiently long first document about gardening tips.",  # exact dup
        "This is a second, sufficiently long and completely distinct document.",
    ]
    result = list(pipeline.run(docs))
    assert result == [
        "This is a sufficiently long first document about gardening tips.",
        "This is a second, sufficiently long and completely distinct document.",
    ]


def test_pipeline_reset_clears_dedup_state() -> None:
    pipeline = ProcessingPipeline([MinHashDeduplicator(MinHashDeduplicatorConfig())])
    docs = ["a repeatable document about clouds and weather patterns"]
    assert list(pipeline.run(docs)) == docs
    pipeline.reset()
    assert list(pipeline.run(docs)) == docs


def test_pipeline_steps_property_reflects_construction_order() -> None:
    normalizer = UnicodeNormalizer(UnicodeNormalizerConfig())
    filt = QualityFilter(QualityFilterConfig())
    pipeline = ProcessingPipeline([normalizer, filt])
    assert pipeline.steps == (normalizer, filt)
