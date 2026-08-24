"""The Evaluator: held-out metrics and zero-shot benchmarks in one place.

Runs a model against held-out data and/or zero-shot benchmarks, producing the
numbers the paper's tables are built from.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from slm_from_scratch.evaluation.benchmark import Benchmark
from slm_from_scratch.evaluation.metrics import Metric
from slm_from_scratch.modeling.base import LanguageModel
from slm_from_scratch.tokenization.base import Tokenizer

__all__ = ["Evaluator"]


class Evaluator:
    """Orchestrates metrics (over a held-out token dataset) and benchmarks (zero-shot).

    Args:
        device: Device to run the model on during evaluation.
        max_batches: Cap the number of batches drawn from a held-out dataset,
            for a bounded-time estimate on a large eval set.
    """

    def __init__(
        self, *, device: torch.device | str = "cpu", max_batches: int | None = None
    ) -> None:
        self.device = torch.device(device)
        self.max_batches = max_batches

    @torch.no_grad()
    def evaluate_metrics(
        self,
        model: LanguageModel,
        dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
        metrics: Sequence[Metric],
        *,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Run ``model`` over ``dataset``, accumulating every metric in ``metrics``.

        Args:
            model: The model to evaluate (moved to :attr:`device`; restored to
                its original mode afterward).
            dataset: A held-out dataset yielding ``(input_ids, target_ids)`` pairs
                (e.g. a :class:`~slm_from_scratch.data.packing.MemmapTokenDataset`
                built from a validation shard directory that was never trained on).
            metrics: The metrics to compute, evaluated together in one pass.
            batch_size: Sequences per batch.

        Returns:
            ``{metric.name: value}`` for every metric in ``metrics``.
        """
        for metric in metrics:
            metric.reset()

        was_training = model.training
        model = model.to(self.device)
        model.eval()

        n = len(dataset)  # type: ignore[arg-type]
        n_batches = (n + batch_size - 1) // batch_size
        if self.max_batches is not None:
            n_batches = min(n_batches, self.max_batches)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            indices = range(start, min(start + batch_size, n))
            inputs = torch.stack([dataset[i][0] for i in indices]).to(self.device)
            targets = torch.stack([dataset[i][1] for i in indices]).to(self.device)
            logits, _ = model(inputs)
            for metric in metrics:
                metric.update(logits=logits, targets=targets)

        if was_training:
            model.train()

        return {metric.name: metric.compute() for metric in metrics}

    @torch.no_grad()
    def evaluate_benchmarks(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        benchmarks: Sequence[Benchmark],
    ) -> dict[str, float]:
        """Run every benchmark in ``benchmarks`` and return ``{benchmark.name: accuracy}``.

        Args:
            model: The model to evaluate (moved to :attr:`device`; restored to
                its original mode afterward).
            tokenizer: Tokenizer used to score benchmark examples.
            benchmarks: The zero-shot benchmarks to run.

        Returns:
            ``{benchmark.name: accuracy}``.
        """
        was_training = model.training
        model = model.to(self.device)
        model.eval()

        results = {
            benchmark.name: benchmark.evaluate(
                model, tokenizer, device=self.device, max_examples=self.max_batches
            )
            for benchmark in benchmarks
        }

        if was_training:
            model.train()
        return results

    @staticmethod
    def estimate_bytes_per_token(
        tokenizer: Tokenizer,
        dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
        *,
        sample_size: int = 200,
    ) -> float:
        """Estimate the average UTF-8 bytes each token decodes to, for :class:`BitsPerByte`.

        Args:
            tokenizer: The tokenizer the dataset was packed with.
            dataset: A packed token dataset to sample from.
            sample_size: Number of examples to sample.

        Returns:
            Average bytes per token over the sample.
        """
        n = min(sample_size, len(dataset))  # type: ignore[arg-type]
        total_bytes = 0
        total_tokens = 0
        for i in range(n):
            input_ids, _ = dataset[i]
            ids = input_ids.tolist()
            text = tokenizer.decode(ids)
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(ids)
        return total_bytes / total_tokens if total_tokens else 1.0
