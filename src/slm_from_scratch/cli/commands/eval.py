"""``slm eval``: evaluate a checkpoint on held-out perplexity and/or zero-shot benchmarks.

Config file shape::

    model:
      vocab_size: 4096
      ...                      # a `model:` section, same shape as a train config

    checkpoint: artifacts/ablation/checkpoints/s6
    tokenizer: artifacts/cli_smoke/tokenizer.json

    held_out:                   # optional: perplexity/bits-per-byte/accuracy on a shard dir
      shard_dir: artifacts/heldout/shards
      batch_size: 32
      max_batches: 50
      metrics: [perplexity, bits_per_byte, token_accuracy]

    benchmarks: [hellaswag, lambada]   # optional: zero-shot benchmark keys
    max_benchmark_examples: 500         # optional cap, for a bounded-time run

    results_json: artifacts/eval/s6.json
    results_latex: paper/tables/s6.tex
    table_caption: "S6 (full modern stack) evaluation"
    table_label: "tab:s6-eval"
"""

from __future__ import annotations

import argparse

import torch

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.cli.util import load_yaml_mapping
from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.data.packing import MemmapTokenDataset
from slm_from_scratch.evaluation import BENCHMARKS, METRICS, Evaluator, ResultsTable
from slm_from_scratch.evaluation.metrics import BitsPerByte, Metric
from slm_from_scratch.modeling import DecoderOnlyTransformer, ModelConfig
from slm_from_scratch.tokenization.base import Tokenizer
from slm_from_scratch.training.checkpoint import CheckpointManager

__all__ = ["EvalCommand"]


@COMMANDS.register("eval")
class EvalCommand(Command):
    """Evaluate a checkpoint: held-out metrics and/or zero-shot benchmarks."""

    name = "eval"
    help = "Evaluate a checkpoint on held-out perplexity and/or zero-shot benchmarks."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register ``--config``."""
        parser.add_argument("--config", required=True, help="Path to an eval YAML config.")

    def run(self, args: argparse.Namespace) -> int:
        """Load the model + checkpoint, run the configured evaluations, save results."""
        spec = load_yaml_mapping(args.config)

        model_config = ModelConfig.from_dict(spec["model"])
        model = DecoderOnlyTransformer(model_config)
        state = CheckpointManager(spec["checkpoint"]).load_latest()
        if state is None:
            raise ConfigError(f"no checkpoint found in {spec['checkpoint']}")
        model.load_state_dict(state.model_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        tokenizer = Tokenizer.load(spec["tokenizer"])
        row: dict[str, float] = {}

        held_out = spec.get("held_out")
        if held_out:
            row.update(self._run_held_out(model, tokenizer, held_out, device))

        benchmark_keys = spec.get("benchmarks") or []
        if benchmark_keys:
            max_examples = spec.get("max_benchmark_examples")
            evaluator_bounded = Evaluator(device=device, max_batches=max_examples)
            benchmarks = [BENCHMARKS.build(key) for key in benchmark_keys]
            row.update(evaluator_bounded.evaluate_benchmarks(model, tokenizer, benchmarks))

        for key, value in row.items():
            print(f"eval: {key} = {value:.4f}")

        table = ResultsTable(
            caption=spec.get("table_caption", "Evaluation results"),
            label=spec.get("table_label", "tab:eval"),
        )
        table.add_row(spec.get("checkpoint", "model"), row)

        if spec.get("results_json"):
            table.save_json(spec["results_json"])
            print(f"eval: results -> {spec['results_json']}")
        if spec.get("results_latex"):
            table.save_latex(spec["results_latex"])
            print(f"eval: LaTeX table -> {spec['results_latex']}")

        return 0

    @staticmethod
    def _run_held_out(
        model: DecoderOnlyTransformer,
        tokenizer: Tokenizer,
        held_out: dict[str, object],
        device: torch.device,
    ) -> dict[str, float]:
        dataset = MemmapTokenDataset(
            str(held_out["shard_dir"]), block_size=model.config.block_size
        )
        metric_keys = held_out.get("metrics") or ["perplexity"]
        assert isinstance(metric_keys, list)

        metrics: list[Metric] = []
        for key in metric_keys:
            if key == "bits_per_byte":
                bpt = Evaluator.estimate_bytes_per_token(tokenizer, dataset)
                metrics.append(BitsPerByte(bytes_per_token=bpt))
            else:
                metrics.append(METRICS.build(key))

        max_batches = held_out.get("max_batches")
        assert max_batches is None or isinstance(max_batches, int)
        evaluator = Evaluator(device=device, max_batches=max_batches)
        batch_size_raw = held_out.get("batch_size", 32)
        assert isinstance(batch_size_raw, int)
        return evaluator.evaluate_metrics(model, dataset, metrics, batch_size=batch_size_raw)
