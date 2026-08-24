"""``slm train-tokenizer``: train and save a tokenizer on a text corpus."""

from __future__ import annotations

import argparse

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.cli.util import load_yaml_mapping, read_corpus
from slm_from_scratch.core.exceptions import ConfigError
from slm_from_scratch.tokenization.base import TOKENIZERS

__all__ = ["TrainTokenizerCommand"]


@COMMANDS.register("train-tokenizer")
class TrainTokenizerCommand(Command):
    """Train a tokenizer on a corpus produced by ``slm prepare-data`` (or plain text lines).

    Config file shape::

        tokenizer:
          type: byte_level_bpe   # a TOKENIZERS registry key
          vocab_size: 4096        # any other TokenizerConfig field
    """

    name = "train-tokenizer"
    help = "Train a tokenizer on a text corpus and save it to disk."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register ``--config``, ``--corpus``, and ``--output``."""
        parser.add_argument(
            "--config", required=True, help="Path to a train-tokenizer YAML config."
        )
        parser.add_argument(
            "--corpus",
            required=True,
            help="Path to a JSONL (prepare-data output) or plain-text corpus.",
        )
        parser.add_argument(
            "--output", required=True, help="Path to save the trained tokenizer JSON."
        )

    def run(self, args: argparse.Namespace) -> int:
        """Train the configured tokenizer on ``--corpus`` and save it to ``--output``."""
        spec = load_yaml_mapping(args.config)
        tok_spec = dict(spec.get("tokenizer") or {})
        kind = tok_spec.pop("type", None)
        if not isinstance(kind, str):
            raise ConfigError("train-tokenizer config needs tokenizer.type")

        tokenizer_cls = TOKENIZERS.get(kind)
        tokenizer = tokenizer_cls(tokenizer_cls.config_cls.from_dict(tok_spec))

        docs = list(read_corpus(args.corpus))
        vocab_target = tok_spec.get("vocab_size")
        print(
            f"train-tokenizer: training {kind!r} (vocab_size target {vocab_target}) "
            f"on {len(docs)} documents ..."
        )
        tokenizer.train(docs)  # type: ignore[attr-defined]
        tokenizer.save(args.output)
        print(f"train-tokenizer: vocab_size={tokenizer.vocab_size} -> {args.output}")
        return 0
