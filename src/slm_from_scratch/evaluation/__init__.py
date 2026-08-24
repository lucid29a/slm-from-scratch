"""Evaluation: metrics over held-out data, zero-shot benchmarks, and results tables.

Importing this package registers every concrete :class:`Metric` and
:class:`Benchmark` in ``METRICS`` / ``BENCHMARKS``.
"""

from __future__ import annotations

from slm_from_scratch.evaluation.benchmark import (
    BENCHMARKS,
    ArcEasy,
    Benchmark,
    HellaSwag,
    Lambada,
    MultipleChoiceBenchmark,
    Piqa,
)
from slm_from_scratch.evaluation.evaluator import Evaluator
from slm_from_scratch.evaluation.metrics import (
    METRICS,
    BitsPerByte,
    Metric,
    Perplexity,
    TokenAccuracy,
)
from slm_from_scratch.evaluation.results_table import ResultsTable

__all__ = [
    "BENCHMARKS",
    "METRICS",
    "ArcEasy",
    "Benchmark",
    "BitsPerByte",
    "Evaluator",
    "HellaSwag",
    "Lambada",
    "Metric",
    "MultipleChoiceBenchmark",
    "Perplexity",
    "Piqa",
    "ResultsTable",
    "TokenAccuracy",
]
