"""Zero-shot benchmarks, scored by likelihood rather than by generating text.

Every benchmark here poses a multiple-choice (or single-continuation) problem
and asks "does the model assign higher likelihood to the correct answer than
the alternatives?" -- no sampling, no generation, no reliance on the model
following an instruction format it was never trained on. This is the standard
way small, non-instruction-tuned base models are evaluated (the same
methodology `lm-evaluation-harness` uses).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

import torch
import torch.nn.functional as F

from slm_from_scratch.core.exceptions import EvaluationError
from slm_from_scratch.core.registry import Registry
from slm_from_scratch.modeling.base import LanguageModel
from slm_from_scratch.tokenization.base import Tokenizer

__all__ = [
    "BENCHMARKS",
    "ArcEasy",
    "Benchmark",
    "HellaSwag",
    "Lambada",
    "MultipleChoiceBenchmark",
    "Piqa",
]


class Benchmark(ABC):
    """Abstract base for a zero-shot evaluation benchmark."""

    name: ClassVar[str]

    @abstractmethod
    def evaluate(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        *,
        device: torch.device,
        max_examples: int | None = None,
    ) -> float:
        """Score ``model`` on this benchmark and return an accuracy in ``[0, 1]``.

        Args:
            model: The model to evaluate, already in eval mode on ``device``.
            tokenizer: Tokenizer used to encode examples.
            device: Device to run scoring on.
            max_examples: Cap the number of examples scored, for a bounded-time run.
        """
        raise NotImplementedError


BENCHMARKS: Registry[Benchmark] = Registry("benchmark", Benchmark)  # type: ignore[type-abstract]


class MultipleChoiceBenchmark(Benchmark, ABC):
    """Shared scoring for "pick the correct continuation among N choices" tasks.

    Each choice is scored by its length-normalized log-likelihood conditioned
    on the shared context (teacher-forced, no sampling); the highest-scoring
    choice is the model's prediction. Length normalization (dividing by the
    number of choice tokens) keeps a long-but-wrong choice from being unfairly
    penalized relative to a short-but-right one purely for having more tokens
    whose probabilities get multiplied together.
    """

    @abstractmethod
    def _iter_examples(self, max_examples: int | None) -> Iterator[tuple[str, list[str], int]]:
        """Yield ``(context, choices, gold_index)`` examples.

        Args:
            max_examples: Stop after this many examples, or ``None`` for all.
        """
        raise NotImplementedError

    def evaluate(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        *,
        device: torch.device,
        max_examples: int | None = None,
    ) -> float:
        """Score every example; return the fraction where the top-scored choice is correct."""
        correct = 0
        total = 0
        for context, choices, gold_index in self._iter_examples(max_examples):
            scores = [
                self._score_choice(model, tokenizer, context, choice, device)
                for choice in choices
            ]
            predicted = max(range(len(scores)), key=lambda i: scores[i])
            correct += int(predicted == gold_index)
            total += 1

        if total == 0:
            raise EvaluationError(f"{self.name}: no examples were scored")
        return correct / total

    @staticmethod
    @torch.no_grad()
    def _score_choice(
        model: LanguageModel,
        tokenizer: Tokenizer,
        context: str,
        choice: str,
        device: torch.device,
    ) -> float:
        """Return the mean per-token log-probability of ``choice`` given ``context``."""
        context_ids = tokenizer.encode(context)
        choice_ids = tokenizer.encode(choice)
        if not choice_ids:
            return float("-inf")

        block_size = model.config.block_size
        full_ids = (context_ids + choice_ids)[-(block_size + 1) :]
        choice_len = min(len(choice_ids), len(full_ids) - 1)

        input_ids = torch.tensor([full_ids[:-1]], device=device)
        target_ids = torch.tensor([full_ids[1:]], device=device)

        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits[0, -choice_len:, :], dim=-1)
        targets = target_ids[0, -choice_len:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.mean().item())


def _require_datasets() -> object:
    try:
        import datasets
    except ImportError as exc:
        raise EvaluationError(
            "this benchmark requires the 'datasets' package; "
            "install with `pip install -e '.[data]'`"
        ) from exc
    return datasets


@BENCHMARKS.register("hellaswag")
class HellaSwag(MultipleChoiceBenchmark):
    """HellaSwag (Zellers et al. 2019): 4-choice commonsense sentence completion."""

    name: ClassVar[str] = "hellaswag"

    def _iter_examples(self, max_examples: int | None) -> Iterator[tuple[str, list[str], int]]:
        datasets = _require_datasets()
        data = datasets.load_dataset("Rowan/hellaswag", split="validation", streaming=True)  # type: ignore[attr-defined]
        for i, example in enumerate(data):
            if max_examples is not None and i >= max_examples:
                return
            yield example["ctx"], list(example["endings"]), int(example["label"])


@BENCHMARKS.register("piqa")
class Piqa(MultipleChoiceBenchmark):
    """PIQA (Bisk et al. 2020): 2-choice physical commonsense reasoning."""

    name: ClassVar[str] = "piqa"

    def _iter_examples(self, max_examples: int | None) -> Iterator[tuple[str, list[str], int]]:
        datasets = _require_datasets()
        data = datasets.load_dataset(  # type: ignore[attr-defined]
            "piqa", split="validation", streaming=True, trust_remote_code=True
        )
        for i, example in enumerate(data):
            if max_examples is not None and i >= max_examples:
                return
            yield example["goal"], [example["sol1"], example["sol2"]], int(example["label"])


@BENCHMARKS.register("arc_easy")
class ArcEasy(MultipleChoiceBenchmark):
    """ARC-Easy (Clark et al. 2018): multiple-choice grade-school science questions."""

    name: ClassVar[str] = "arc_easy"

    def _iter_examples(self, max_examples: int | None) -> Iterator[tuple[str, list[str], int]]:
        datasets = _require_datasets()
        data = datasets.load_dataset(  # type: ignore[attr-defined]
            "allenai/ai2_arc", "ARC-Easy", split="test", streaming=True
        )
        for i, example in enumerate(data):
            if max_examples is not None and i >= max_examples:
                return
            labels = list(example["choices"]["label"])
            texts = list(example["choices"]["text"])
            answer = example["answerKey"]
            if answer not in labels:
                continue
            yield example["question"], texts, labels.index(answer)


@BENCHMARKS.register("lambada")
class Lambada(Benchmark):
    """LAMBADA (Paperno et al. 2016): predict a passage's final word from its context.

    Unlike the multiple-choice benchmarks above, LAMBADA has no distractors --
    accuracy is whether the model's greedy next-token prediction after the
    context matches the true final word's first token.
    """

    name: ClassVar[str] = "lambada"

    def evaluate(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        *,
        device: torch.device,
        max_examples: int | None = None,
    ) -> float:
        """Return the fraction of examples where greedy decoding predicts the target word."""
        datasets = _require_datasets()
        data = datasets.load_dataset(  # type: ignore[attr-defined]
            "EleutherAI/lambada_openai", "default", split="test", streaming=True
        )

        correct, total = 0, 0
        for i, example in enumerate(data):
            if max_examples is not None and i >= max_examples:
                break
            text = example["text"]
            context, _, target_word = text.rpartition(" ")
            if not context or not target_word:
                continue

            target_ids = tokenizer.encode(" " + target_word)
            if not target_ids:
                continue

            block_size = model.config.block_size
            context_ids = tokenizer.encode(context)[-block_size:]
            input_ids = torch.tensor([context_ids], device=device)
            with torch.no_grad():
                logits, _ = model(input_ids)
            predicted_id = int(logits[0, -1].argmax().item())

            correct += int(predicted_id == target_ids[0])
            total += 1

        if total == 0:
            raise EvaluationError("lambada: no examples were scored")
        return correct / total
