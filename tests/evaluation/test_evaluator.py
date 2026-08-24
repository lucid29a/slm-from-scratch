"""Tests for Evaluator."""

from __future__ import annotations

import torch

from slm_from_scratch.evaluation.evaluator import Evaluator
from slm_from_scratch.evaluation.metrics import Perplexity, TokenAccuracy
from tests.evaluation.conftest import SyntheticLMDataset, make_tiny_model


def test_evaluate_metrics_returns_all_requested_metrics() -> None:
    model = make_tiny_model()
    dataset = SyntheticLMDataset(200, block_size=16, vocab_size=50)
    evaluator = Evaluator(device="cpu")
    metrics = [Perplexity(), TokenAccuracy()]
    results = evaluator.evaluate_metrics(model, dataset, metrics, batch_size=8)
    assert set(results) == {"perplexity", "token_accuracy"}
    assert results["perplexity"] > 0
    assert 0.0 <= results["token_accuracy"] <= 1.0


def test_evaluate_metrics_restores_model_training_mode() -> None:
    model = make_tiny_model()
    model.train()
    dataset = SyntheticLMDataset(50, block_size=16, vocab_size=50)
    Evaluator(device="cpu").evaluate_metrics(model, dataset, [Perplexity()], batch_size=8)
    assert model.training is True


def test_evaluate_metrics_leaves_eval_mode_as_eval() -> None:
    model = make_tiny_model()
    model.eval()
    dataset = SyntheticLMDataset(50, block_size=16, vocab_size=50)
    Evaluator(device="cpu").evaluate_metrics(model, dataset, [Perplexity()], batch_size=8)
    assert model.training is False


def test_max_batches_bounds_the_number_of_batches_scored() -> None:
    model = make_tiny_model()
    dataset = SyntheticLMDataset(1000, block_size=16, vocab_size=50)

    class CountingMetric(TokenAccuracy):
        def __init__(self) -> None:
            super().__init__()
            self.update_calls = 0

        def update(self, *, logits: torch.Tensor, targets: torch.Tensor) -> None:
            self.update_calls += 1
            super().update(logits=logits, targets=targets)

    metric = CountingMetric()
    Evaluator(device="cpu", max_batches=3).evaluate_metrics(model, dataset, [metric], batch_size=8)
    assert metric.update_calls == 3


def test_metrics_reset_before_each_evaluate_call() -> None:
    model = make_tiny_model()
    dataset = SyntheticLMDataset(200, block_size=16, vocab_size=50)
    metric = TokenAccuracy()
    evaluator = Evaluator(device="cpu")
    evaluator.evaluate_metrics(model, dataset, [metric], batch_size=8)
    first_total = metric._total
    evaluator.evaluate_metrics(model, dataset, [metric], batch_size=8)
    second_total = metric._total
    assert first_total == second_total  # reset, not accumulated across calls


def test_estimate_bytes_per_token_is_positive() -> None:
    from slm_from_scratch.tokenization.char import CharTokenizer, CharTokenizerConfig

    tokenizer = CharTokenizer(CharTokenizerConfig(vocab_size=64)).train(
        ["the quick brown fox jumps over the lazy dog"]
    )
    dataset = SyntheticLMDataset(50, block_size=16, vocab_size=tokenizer.vocab_size)
    bpt = Evaluator.estimate_bytes_per_token(tokenizer, dataset, sample_size=20)
    assert bpt > 0
