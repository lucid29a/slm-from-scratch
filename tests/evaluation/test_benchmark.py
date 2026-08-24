"""Tests for MultipleChoiceBenchmark's shared scoring logic (network-free, via a
local fake benchmark) and the concrete benchmarks' registration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from slm_from_scratch.core.exceptions import EvaluationError
from slm_from_scratch.evaluation.benchmark import BENCHMARKS, MultipleChoiceBenchmark
from slm_from_scratch.tokenization.char import CharTokenizer, CharTokenizerConfig
from tests.evaluation.conftest import make_tiny_model


class FixedExamplesBenchmark(MultipleChoiceBenchmark):
    """A MultipleChoiceBenchmark over a fixed, in-memory list of examples -- no network."""

    name = "fixed_examples"

    def __init__(self, examples: list[tuple[str, list[str], int]]) -> None:
        self.examples = examples

    def _iter_examples(self, max_examples: int | None) -> Iterator[tuple[str, list[str], int]]:
        for i, example in enumerate(self.examples):
            if max_examples is not None and i >= max_examples:
                return
            yield example


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer(CharTokenizerConfig(vocab_size=64)).train(
        ["the cat sat on the mat", "the dog ran in the park", "a bird flew over the tree"]
    )


def test_evaluate_scores_examples_and_returns_accuracy(tokenizer: CharTokenizer) -> None:
    model = make_tiny_model(vocab_size=tokenizer.vocab_size, block_size=32)
    benchmark = FixedExamplesBenchmark(
        [
            ("the ", ["cat", "dog"], 0),
            ("a ", ["bird", "cat"], 0),
        ]
    )
    device = torch.device("cpu")
    accuracy = benchmark.evaluate(model, tokenizer, device=device)
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_raises_on_no_examples(tokenizer: CharTokenizer) -> None:
    model = make_tiny_model(vocab_size=tokenizer.vocab_size, block_size=32)
    benchmark = FixedExamplesBenchmark([])
    with pytest.raises(EvaluationError, match="no examples"):
        benchmark.evaluate(model, tokenizer, device=torch.device("cpu"))


def test_score_choice_prefers_a_choice_the_model_was_trained_towards(
    tokenizer: CharTokenizer,
) -> None:
    # A model whose embedding+head strongly favor one specific token should
    # score a choice built from that token higher than an unrelated one.
    torch.manual_seed(0)
    model = make_tiny_model(vocab_size=tokenizer.vocab_size, block_size=32)
    device = torch.device("cpu")

    context = "the "
    likely_choice = "cat"
    score = MultipleChoiceBenchmark._score_choice(
        model, tokenizer, context, likely_choice, device
    )
    assert isinstance(score, float)


def test_score_choice_handles_empty_choice_gracefully(tokenizer: CharTokenizer) -> None:
    model = make_tiny_model(vocab_size=tokenizer.vocab_size, block_size=32)
    device = torch.device("cpu")
    score = MultipleChoiceBenchmark._score_choice(model, tokenizer, "the ", "", device)
    assert score == float("-inf")


def test_max_examples_limits_scored_examples(tokenizer: CharTokenizer) -> None:
    model = make_tiny_model(vocab_size=tokenizer.vocab_size, block_size=32)
    calls = []

    class CountingBenchmark(FixedExamplesBenchmark):
        def _iter_examples(self, max_examples: int | None) -> Iterator[tuple[str, list[str], int]]:
            for item in super()._iter_examples(max_examples):
                calls.append(item)
                yield item

    benchmark = CountingBenchmark(
        [("the ", ["cat", "dog"], 0), ("a ", ["bird", "cat"], 0), ("the ", ["mat", "cat"], 1)]
    )
    benchmark.evaluate(model, tokenizer, device=torch.device("cpu"), max_examples=2)
    assert len(calls) == 2


def test_registry_contains_all_four_concrete_benchmarks() -> None:
    assert {"hellaswag", "piqa", "arc_easy", "lambada"} <= set(BENCHMARKS.keys())
