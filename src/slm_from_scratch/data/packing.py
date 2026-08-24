"""Turning a tokenizer + a document stream into fixed-size training examples on disk.

Three responsibilities, three classes:

* :class:`SequencePacker` -- encode documents and concatenate their token ids into
  one flat stream, separated by ``<eos>``. This is what lets a training batch mix
  the tail of one document with the head of the next instead of padding every
  document out to a fixed length (padding wastes compute; packing doesn't).
* :class:`BinaryShardWriter` -- write that flat stream to disk as fixed-size
  ``uint16``/``uint32`` binary shards plus a JSON manifest, so a corpus far larger
  than RAM (a 3B-token FineWeb-Edu sample is ~6 GB as uint16) can be packed once
  and then memory-mapped cheaply on every subsequent training run.
* :class:`MemmapTokenDataset` -- a :class:`torch.utils.data.Dataset` that reads
  those shards back, handing the trainer ``(input_ids, target_ids)`` pairs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from slm_from_scratch.core.exceptions import DataError
from slm_from_scratch.tokenization.base import Tokenizer

__all__ = ["BinaryShardWriter", "MemmapTokenDataset", "SequencePacker", "ShardManifest"]

_MANIFEST_NAME = "manifest.json"


class SequencePacker:
    """Encodes documents and concatenates them into one flat token stream.

    Args:
        tokenizer: The tokenizer used to encode each document.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def pack(self, documents: Iterable[str]) -> Iterator[int]:
        """Yield token ids: each document's ids followed by a single ``<eos>``.

        Args:
            documents: The documents to encode, in order.

        Yields:
            Token ids, one at a time, ready to be written to a shard.
        """
        eos_id = self._tokenizer.eos_id
        for doc in documents:
            yield from self._tokenizer.encode(doc)
            yield eos_id


@dataclass(frozen=True, kw_only=True)
class ShardManifest:
    """On-disk record of a packed corpus: shard filenames, dtype, and sizes.

    Attributes:
        shard_files: Shard filenames, in order, relative to the manifest.
        shard_lengths: Number of tokens in each shard, same order as ``shard_files``.
        dtype: Numpy dtype name the shards were written with (``"uint16"`` or
            ``"uint32"``, chosen to fit the tokenizer's vocabulary).
        vocab_size: The tokenizer's vocabulary size at packing time.
    """

    shard_files: list[str]
    shard_lengths: list[int]
    dtype: str
    vocab_size: int

    @property
    def total_tokens(self) -> int:
        """Total token count across all shards."""
        return sum(self.shard_lengths)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "shard_files": self.shard_files,
            "shard_lengths": self.shard_lengths,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardManifest:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            shard_files=list(data["shard_files"]),
            shard_lengths=list(data["shard_lengths"]),
            dtype=data["dtype"],
            vocab_size=data["vocab_size"],
        )


class BinaryShardWriter:
    """Writes a flat token-id stream to fixed-size binary shards plus a manifest.

    Args:
        out_dir: Directory to write shards and ``manifest.json`` into.
        vocab_size: The tokenizer's vocabulary size; determines whether shards
            are written as ``uint16`` (vocab fits in 16 bits) or ``uint32``.
        tokens_per_shard: Maximum number of tokens per shard file.
        shard_prefix: Filename prefix for shard files, e.g. ``"train"`` ->
            ``train_00000.bin``, ``train_00001.bin``, ...
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        vocab_size: int,
        tokens_per_shard: int = 10_000_000,
        shard_prefix: str = "shard",
    ) -> None:
        if tokens_per_shard <= 0:
            raise DataError(f"tokens_per_shard must be positive, got {tokens_per_shard}")
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._vocab_size = vocab_size
        self._dtype = np.uint16 if vocab_size <= 2**16 - 1 else np.uint32
        self._tokens_per_shard = tokens_per_shard
        self._shard_prefix = shard_prefix

    def write(self, token_stream: Iterable[int]) -> ShardManifest:
        """Consume ``token_stream``, writing it out as shards, and return the manifest.

        Args:
            token_stream: A flat stream of token ids, e.g. from :class:`SequencePacker`.

        Returns:
            The manifest describing what was written; also saved as
            ``manifest.json`` in ``out_dir``.
        """
        shard_files: list[str] = []
        shard_lengths: list[int] = []
        buffer: list[int] = []

        def flush(shard_index: int) -> None:
            if not buffer:
                return
            name = f"{self._shard_prefix}_{shard_index:05d}.bin"
            array = np.array(buffer, dtype=self._dtype)
            array.tofile(self._out_dir / name)
            shard_files.append(name)
            shard_lengths.append(len(array))
            buffer.clear()

        shard_index = 0
        for token_id in token_stream:
            buffer.append(token_id)
            if len(buffer) >= self._tokens_per_shard:
                flush(shard_index)
                shard_index += 1
        flush(shard_index)

        manifest = ShardManifest(
            shard_files=shard_files,
            shard_lengths=shard_lengths,
            dtype=np.dtype(self._dtype).name,
            vocab_size=self._vocab_size,
        )
        (self._out_dir / _MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        return manifest


class MemmapTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Reads packed shards back as ``(input_ids, target_ids)`` language-modeling pairs.

    Each shard is memory-mapped independently; valid start positions within a
    shard stop ``block_size`` tokens before its end, so no example reads across a
    shard boundary. On a corpus packed into a handful of large shards this costs
    a negligible fraction of the usable positions.

    Args:
        shard_dir: Directory containing ``manifest.json`` and its shard files.
        block_size: Number of tokens per training example.
    """

    def __init__(self, shard_dir: str | Path, *, block_size: int) -> None:
        if block_size <= 0:
            raise DataError(f"block_size must be positive, got {block_size}")
        self._shard_dir = Path(shard_dir)
        manifest_path = self._shard_dir / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise DataError(f"no manifest.json found in {self._shard_dir}")

        self._manifest = ShardManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        self._block_size = block_size
        self._dtype = np.dtype(self._manifest.dtype)

        self._shards: list[np.memmap] = [
            np.memmap(self._shard_dir / name, dtype=self._dtype, mode="r")
            for name in self._manifest.shard_files
        ]

        # Cumulative count of valid start-positions per shard, for O(log n) lookup.
        self._valid_per_shard = [max(len(s) - block_size, 0) for s in self._shards]
        self._cumulative = np.cumsum([0, *self._valid_per_shard])

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for dataset of length {len(self)}")

        shard_idx = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        local_idx = index - int(self._cumulative[shard_idx])
        shard = self._shards[shard_idx]

        chunk = shard[local_idx : local_idx + self._block_size + 1].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1].copy())
        target_ids = torch.from_numpy(chunk[1:].copy())
        return input_ids, target_ids

    @property
    def manifest(self) -> ShardManifest:
        """The manifest this dataset was loaded from."""
        return self._manifest
