"""``slm prepare-data``: source -> processing pipeline -> a flat text corpus file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.cli.util import load_yaml_mapping
from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.data.base import SOURCES
from slm_from_scratch.data.processing import PROCESSING_STEPS, ProcessingPipeline

__all__ = ["PrepareDataCommand"]


@COMMANDS.register("prepare-data")
class PrepareDataCommand(Command):
    """Run a source through a cleaning pipeline and write the surviving documents to disk.

    Config file shape::

        source:
          type: tinystories      # a SOURCES registry key
          limit: 20000            # any other TextSourceConfig field
        processing:               # a list of PROCESSING_STEPS specs, run in order
          - type: unicode_normalize
          - type: quality_filter
            min_chars: 32
          - type: minhash_dedup
    """

    name = "prepare-data"
    help = "Run a text source through a cleaning pipeline, writing one document per line."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register ``--config`` and ``--output``."""
        parser.add_argument("--config", required=True, help="Path to a prepare-data YAML config.")
        parser.add_argument(
            "--output", required=True, help="Output path (JSONL, one doc per line)."
        )

    def run(self, args: argparse.Namespace) -> int:
        """Load the config, run the pipeline, and write output as JSONL."""
        spec = load_yaml_mapping(args.config)

        source_spec = dict(spec.get("source") or {})
        source_kind = source_spec.pop("type", None)
        if not isinstance(source_kind, str):
            raise ConfigError("prepare-data config needs source.type")
        source_cls = SOURCES.get(source_kind)
        source = source_cls(source_cls.config_cls.from_dict(source_spec))

        steps = []
        for step_spec in spec.get("processing") or []:
            step_spec = dict(step_spec)
            step_kind = step_spec.pop("type", None)
            if not isinstance(step_kind, str):
                raise ConfigError("each processing entry needs a 'type'")
            step_cls = PROCESSING_STEPS.get(step_kind)
            steps.append(step_cls(step_cls.config_cls.from_dict(step_spec)))
        pipeline = ProcessingPipeline(steps)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        counts = {"in": 0, "out": 0}

        def counted_source() -> Any:
            for doc in source:
                counts["in"] += 1
                yield doc

        with out_path.open("w", encoding="utf-8") as f:
            for cleaned in pipeline.run(counted_source()):
                f.write(json.dumps({"text": cleaned}, ensure_ascii=False))
                f.write("\n")
                counts["out"] += 1

        print(f"prepare-data: {counts['in']} documents read, {counts['out']} kept -> {out_path}")
        return 0
