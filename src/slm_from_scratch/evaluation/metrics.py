"""Metrics: accumulate over batches, report a scalar.

The building blocks of :class:`~slm_from_scratch.evaluation.evaluator.Evaluator`'s
held-out numbers.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn.functional as F

from slm_from_scratch.core.exceptions import EvaluationError
from slm_from_scratch.core.registry import Registry

__all__ = ["METRICS", "BitsPerByte", "Metric", "Perplexity", "TokenAccuracy"]


class Metric(ABC):
    """Abstract base for a streaming, batch-accumulating evaluation metric.

    A metric never sees a whole dataset at once -- :meth:`update` is called
    once per batch, and :meth:`compute` derives the final scalar from whatever
    running totals ``update`` accumulated. This is what lets
    :class:`~slm_from_scratch.evaluation.evaluator.Evaluator` stream over an
    arbitrarily large held-out set without holding it all in memory.
    """

    #: Registry key / display name, set by subclasses.
    name: ClassVar[str]

    @abstractmethod
    def update(self, *, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate statistics from one batch.

        Args:
            logits: ``(batch, seq_len, vocab_size)`` unnormalized next-token scores.
            targets: ``(batch, seq_len)`` gold next-token ids.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """Return the metric's value over everything seen since the last :meth:`reset`.

        Raises:
            EvaluationError: If no batches have been accumulated yet.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state."""
        raise NotImplementedError


METRICS: Registry[Metric] = Registry("metric", Metric)  # type: ignore[type-abstract]


class _CrossEntropyAccumulator(Metric, ABC):
    """Shared bookkeeping for metrics derived from summed token-level cross-entropy."""

    def __init__(self) -> None:
        self._sum_nll = 0.0
        self._count = 0

    def update(self, *, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate summed negative log-likelihood and token count for this batch."""
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
        )
        self._sum_nll += float(loss.item())
        self._count += targets.numel()

    def reset(self) -> None:
        """Clear accumulated totals."""
        self._sum_nll = 0.0
        self._count = 0

    def _mean_nll_nats(self) -> float:
        if self._count == 0:
            raise EvaluationError(f"{self.name}.compute() called before any update()")
        return self._sum_nll / self._count


@METRICS.register("perplexity")
class Perplexity(_CrossEntropyAccumulator):
    """Exponential of the mean per-token cross-entropy: ``exp(mean_nll)``.

    The standard held-out language-modeling metric -- lower is better, and it
    has the intuitive reading "the model is, on average, as uncertain as if it
    were choosing uniformly among this many tokens."
    """

    name: ClassVar[str] = "perplexity"

    def compute(self) -> float:
        """Return ``exp(mean negative log-likelihood in nats)``."""
        return math.exp(self._mean_nll_nats())


@METRICS.register("bits_per_byte")
class BitsPerByte(_CrossEntropyAccumulator):
    """Mean cross-entropy converted to bits per *byte* of original text.

    Comparable across tokenizers with different vocabularies/compression
    ratios (unlike perplexity, which is measured per *token* and so isn't
    comparable between a coarse and a fine tokenizer). Requires knowing the
    average number of bytes each token decodes to, supplied at construction.

    Args:
        bytes_per_token: Average number of UTF-8 bytes per token in the
            evaluation set (see
            :meth:`~slm_from_scratch.evaluation.evaluator.Evaluator.estimate_bytes_per_token`).
    """

    name: ClassVar[str] = "bits_per_byte"

    def __init__(self, bytes_per_token: float) -> None:
        super().__init__()
        if bytes_per_token <= 0:
            raise EvaluationError(f"bytes_per_token must be positive, got {bytes_per_token}")
        self.bytes_per_token = bytes_per_token

    def compute(self) -> float:
        """Return ``(mean_nll_nats / ln(2)) / bytes_per_token``."""
        mean_nll_bits = self._mean_nll_nats() / math.log(2)
        return mean_nll_bits / self.bytes_per_token


@METRICS.register("token_accuracy")
class TokenAccuracy(Metric):
    """Fraction of positions where the argmax prediction equals the gold next token."""

    name: ClassVar[str] = "token_accuracy"

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, *, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate correct/total counts for this batch."""
        predictions = logits.argmax(dim=-1)
        self._correct += int((predictions == targets).sum().item())
        self._total += targets.numel()

    def compute(self) -> float:
        """Return ``correct / total``."""
        if self._total == 0:
            raise EvaluationError("TokenAccuracy.compute() called before any update()")
        return self._correct / self._total

    def reset(self) -> None:
        """Clear accumulated counts."""
        self._correct = 0
        self._total = 0
