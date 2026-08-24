"""``slm ablate``: run every training config in a directory, sequentially.

Built for the paper's ablation ladder -- seven configs (``s0_vanilla.yaml``
through ``s6_full.yaml``), each changing exactly one architectural choice from
the previous, all sharing a token budget. Each config is a normal ``slm train``
config (see :mod:`slm_from_scratch.cli.commands.train`); this command's only
job is to run them one after another and collect a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.cli.commands.train import build_trainer
from slm_from_scratch.cli.util import load_yaml_mapping
from slm_from_scratch.training.callback import Callback, StepMetrics
from slm_from_scratch.training.trainer import Trainer

__all__ = ["AblateCommand"]


class _FinalLossRecorder(Callback):
    def __init__(self) -> None:
        self.last_loss: float | None = None

    def on_step_end(self, trainer: Trainer, metrics: StepMetrics) -> None:  # noqa: ARG002
        self.last_loss = metrics.loss


@COMMANDS.register("ablate")
class AblateCommand(Command):
    """Run every ``*.yaml`` training config in a directory and summarize final loss."""

    name = "ablate"
    help = "Run a directory of training configs sequentially (the ablation ladder)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register ``--config-dir`` and ``--summary``."""
        parser.add_argument(
            "--config-dir", required=True, help="Directory of training YAML configs."
        )
        parser.add_argument(
            "--summary", required=True, help="Where to write the JSON summary of final losses."
        )

    def run(self, args: argparse.Namespace) -> int:
        """Train every config in ``--config-dir``, in filename order, and write a summary."""
        config_dir = Path(args.config_dir)
        config_paths = sorted(config_dir.glob("*.yaml"))
        if not config_paths:
            raise SystemExit(f"no *.yaml configs found in {config_dir}")

        results: dict[str, dict[str, object]] = {}
        for path in config_paths:
            print(f"\n=== ablate: {path.name} ===")
            spec = load_yaml_mapping(path)
            trainer = build_trainer(spec)
            recorder = _FinalLossRecorder()
            trainer.callbacks.append(recorder)
            trainer.train()
            results[path.stem] = {
                "final_step": trainer.current_step,
                "final_loss": recorder.last_loss,
            }

        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nablate: summary -> {args.summary}")
        for name, result in results.items():
            print(f"  {name}: final_loss={result['final_loss']}")
        return 0
