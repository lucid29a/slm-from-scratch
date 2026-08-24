"""``slm pack``: tokenize a corpus and write it out as packed binary shards."""

from __future__ import annotations

import argparse

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.cli.util import read_corpus
from slm_from_scratch.data.packing import BinaryShardWriter, SequencePacker
from slm_from_scratch.tokenization.base import Tokenizer

__all__ = ["PackCommand"]


@COMMANDS.register("pack")
class PackCommand(Command):
    """Encode a corpus with a trained tokenizer and write fixed-size binary shards."""

    name = "pack"
    help = "Tokenize a corpus and write packed binary shards for training."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register corpus/tokenizer/output arguments."""
        parser.add_argument(
            "--corpus", required=True, help="JSONL (prepare-data) or plain-text corpus."
        )
        parser.add_argument(
            "--tokenizer", required=True, help="Path to a tokenizer saved by save()."
        )
        parser.add_argument(
            "--output", required=True, help="Directory to write shards + manifest.json to."
        )
        parser.add_argument(
            "--tokens-per-shard", type=int, default=20_000_000, help="Max tokens per shard file."
        )

    def run(self, args: argparse.Namespace) -> int:
        """Pack ``--corpus`` with ``--tokenizer`` into ``--output``."""
        tokenizer = Tokenizer.load(args.tokenizer)
        packer = SequencePacker(tokenizer)
        writer = BinaryShardWriter(
            args.output, vocab_size=tokenizer.vocab_size, tokens_per_shard=args.tokens_per_shard
        )
        manifest = writer.write(packer.pack(read_corpus(args.corpus)))
        print(
            f"pack: {manifest.total_tokens:,} tokens across {len(manifest.shard_files)} "
            f"shard(s) ({manifest.dtype}) -> {args.output}"
        )
        return 0
