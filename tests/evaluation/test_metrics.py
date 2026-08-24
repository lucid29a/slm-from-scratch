"""Tests for Perplexity, BitsPerByte, and TokenAccuracy."""

from __future__ import annotations

import math

import pytest
import torch

from slm_from_scratch.core.exceptions import EvaluationError
from slm_from_scratch.evaluation.metrics import METRICS, BitsPerByte, Perplexity, TokenAccuracy


def _perfect_logits(
    targets: torch.Tensor, vocab_size: int, *, confidence: float = 20.0
) -> torch.Tensor:
    """Build logits that overwhelmingly favor the correct target at every position."""
    logits = torch.zeros(*targets.shape, vocab_size)
    logits.scatter_(-1, targets.unsqueeze(-1), confidence)
    return logits


def test_perplexity_is_low_for_confident_correct_predictions() -> None:
    targets = torch.randint(0, 20, (2, 8))
    logits = _perfect_logits(targets, vocab_size=20)
    metric = Perplexity()
    metric.update(logits=logits, targets=targets)
    assert metric.compute() < 1.01


def test_perplexity_is_roughly_vocab_size_for_uniform_logits() -> None:
    vocab_size = 100
    targets = torch.randint(0, vocab_size, (4, 16))
    logits = torch.zeros(4, 16, vocab_size)  # uniform distribution
    metric = Perplexity()
    metric.update(logits=logits, targets=targets)
    assert metric.compute() == pytest.approx(vocab_size, rel=0.05)


def test_perplexity_accumulates_across_batches() -> None:
    metric = Perplexity()
    t1 = torch.randint(0, 20, (2, 8))
    t2 = torch.randint(0, 20, (2, 8))
    metric.update(logits=_perfect_logits(t1, 20), targets=t1)
    metric.update(logits=_perfect_logits(t2, 20), targets=t2)
    assert metric.compute() < 1.01


def test_perplexity_reset_clears_state() -> None:
    metric = Perplexity()
    targets = torch.randint(0, 20, (2, 8))
    metric.update(logits=_perfect_logits(targets, 20), targets=targets)
    metric.reset()
    with pytest.raises(EvaluationError):
        metric.compute()


def test_token_accuracy_perfect_predictions() -> None:
    targets = torch.randint(0, 20, (2, 8))
    logits = _perfect_logits(targets, vocab_size=20)
    metric = TokenAccuracy()
    metric.update(logits=logits, targets=targets)
    assert metric.compute() == 1.0


def test_token_accuracy_all_wrong() -> None:
    targets = torch.zeros(2, 8, dtype=torch.long)
    logits = torch.zeros(2, 8, 5)
    logits[..., 1] = 10.0  # argmax always predicts class 1, never the target class 0
    metric = TokenAccuracy()
    metric.update(logits=logits, targets=targets)
    assert metric.compute() == 0.0


def test_token_accuracy_before_update_raises() -> None:
    with pytest.raises(EvaluationError):
        TokenAccuracy().compute()


def test_bits_per_byte_scales_inversely_with_bytes_per_token() -> None:
    targets = torch.randint(0, 20, (2, 8))
    logits = _perfect_logits(targets, vocab_size=20, confidence=2.0)  # imperfect, nonzero loss

    low_bpt = BitsPerByte(bytes_per_token=1.0)
    high_bpt = BitsPerByte(bytes_per_token=4.0)
    low_bpt.update(logits=logits, targets=targets)
    high_bpt.update(logits=logits, targets=targets)

    assert high_bpt.compute() == pytest.approx(low_bpt.compute() / 4.0)


def test_bits_per_byte_matches_hand_computed_value() -> None:
    vocab_size = 8
    targets = torch.zeros(1, 1, dtype=torch.long)
    logits = torch.zeros(1, 1, vocab_size)  # uniform -> nll = ln(vocab_size)
    metric = BitsPerByte(bytes_per_token=2.0)
    metric.update(logits=logits, targets=targets)
    expected = (math.log(vocab_size) / math.log(2)) / 2.0
    assert metric.compute() == pytest.approx(expected)


def test_bits_per_byte_rejects_nonpositive_bytes_per_token() -> None:
    with pytest.raises(EvaluationError, match="bytes_per_token"):
        BitsPerByte(bytes_per_token=0.0)


def test_registry_contains_all_three() -> None:
    assert "perplexity" in METRICS
    assert "token_accuracy" in METRICS
    assert "bits_per_byte" in METRICS
