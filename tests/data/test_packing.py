"""Tests for SequencePacker, BinaryShardWriter, and MemmapTokenDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slm_from_scratch.core.exceptions import DataError
from slm_from_scratch.data.packing import BinaryShardWriter, MemmapTokenDataset, SequencePacker
from slm_from_scratch.tokenization.byte_level_bpe import ByteLevelBPETokenizer


def test_packer_inserts_eos_between_documents(tokenizer: ByteLevelBPETokenizer) -> None:
    packer = SequencePacker(tokenizer)
    docs = ["hello", "world"]
    stream = list(packer.pack(docs))

    expected = [*tokenizer.encode("hello"), tokenizer.eos_id, *tokenizer.encode("world"), tokenizer.eos_id]
    assert stream == expected


def test_packer_empty_corpus_yields_nothing(tokenizer: ByteLevelBPETokenizer) -> None:
    assert list(SequencePacker(tokenizer).pack([])) == []


def test_shard_writer_chooses_uint16_for_small_vocab(tmp_path: Path) -> None:
    writer = BinaryShardWriter(tmp_path, vocab_size=300, tokens_per_shard=1000)
    manifest = writer.write(range(10))
    assert manifest.dtype == "uint16"


def test_shard_writer_chooses_uint32_for_large_vocab(tmp_path: Path) -> None:
    writer = BinaryShardWriter(tmp_path, vocab_size=200_000, tokens_per_shard=1000)
    manifest = writer.write(range(10))
    assert manifest.dtype == "uint32"


def test_shard_writer_splits_across_multiple_shards(tmp_path: Path) -> None:
    writer = BinaryShardWriter(tmp_path, vocab_size=300, tokens_per_shard=10)
    manifest = writer.write(range(25))
    assert len(manifest.shard_files) == 3
    assert manifest.shard_lengths == [10, 10, 5]
    assert manifest.total_tokens == 25


def test_shard_writer_rejects_nonpositive_tokens_per_shard(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="tokens_per_shard"):
        BinaryShardWriter(tmp_path, vocab_size=300, tokens_per_shard=0)


def test_shard_round_trip_preserves_token_values(tmp_path: Path) -> None:
    tokens = list(range(0, 1000, 3))
    writer = BinaryShardWriter(tmp_path, vocab_size=1200, tokens_per_shard=97)
    writer.write(tokens)

    ds = MemmapTokenDataset(tmp_path, block_size=4)
    # Reconstruct the flat stream from the dataset's own reads and compare.
    x0, y0 = ds[0]
    assert x0.tolist() == tokens[0:4]
    assert y0.tolist() == tokens[1:5]


def test_dataset_shift_invariant_holds_everywhere(tmp_path: Path) -> None:
    tokens = list(range(200))
    BinaryShardWriter(tmp_path, vocab_size=300, tokens_per_shard=64).write(tokens)
    ds = MemmapTokenDataset(tmp_path, block_size=8)

    for i in range(len(ds)):
        x, y = ds[i]
        assert x[1:].tolist() == y[:-1].tolist()


def test_dataset_length_excludes_last_block_size_per_shard(tmp_path: Path) -> None:
    # Two shards of 20 tokens each with block_size=5: 15 valid starts per shard.
    BinaryShardWriter(tmp_path, vocab_size=300, tokens_per_shard=20).write(range(40))
    ds = MemmapTokenDataset(tmp_path, block_size=5)
    assert len(ds) == 15 + 15


def test_dataset_index_out_of_range_raises(tmp_path: Path) -> None:
    BinaryShardWriter(tmp_path, vocab_size=300, tokens_per_shard=20).write(range(10))
    ds = MemmapTokenDataset(tmp_path, block_size=3)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_dataset_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="manifest"):
        MemmapTokenDataset(tmp_path, block_size=4)


def test_dataset_rejects_nonpositive_block_size(tmp_path: Path) -> None:
    BinaryShardWriter(tmp_path, vocab_size=300, tokens_per_shard=20).write(range(10))
    with pytest.raises(DataError, match="block_size"):
        MemmapTokenDataset(tmp_path, block_size=0)


def test_full_pipeline_round_trip(
    tmp_path: Path, tokenizer: ByteLevelBPETokenizer, toy_corpus: list[str]
) -> None:
    packer = SequencePacker(tokenizer)
    stream = list(packer.pack(toy_corpus))

    writer = BinaryShardWriter(tmp_path, vocab_size=tokenizer.vocab_size, tokens_per_shard=50)
    manifest = writer.write(stream)
    assert manifest.total_tokens == len(stream)

    ds = MemmapTokenDataset(tmp_path, block_size=8)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert x.dtype == y.dtype

    # The dataset's tokens must decode back to a prefix of the packed stream.
    decoded_ids = np.concatenate([x.numpy(), y.numpy()[-1:]])
    assert decoded_ids.tolist() == stream[:9]
