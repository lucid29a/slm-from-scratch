"""End-to-end, network-free tests of the CLI pipeline: prepare-data -> train-tokenizer
-> pack -> train -> generate, all driven off local files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from slm_from_scratch.cli.commands.ablate import AblateCommand
from slm_from_scratch.cli.commands.generate import GenerateCommand
from slm_from_scratch.cli.commands.pack import PackCommand
from slm_from_scratch.cli.commands.prepare_data import PrepareDataCommand
from slm_from_scratch.cli.commands.train import TrainCommand
from slm_from_scratch.cli.commands.train_tokenizer import TrainTokenizerCommand

_STORIES = [
    "The quick brown fox jumps over the lazy dog in the meadow every morning.",
    "A curious cat watches birds from the windowsill every single day.",
    "Rain fell softly on the old wooden roof throughout the quiet night.",
    "The little robot learned to paint bright pictures of the sunset.",
    "Two children built a sandcastle by the shore before the tide came in.",
]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    src = tmp_path / "raw"
    src.mkdir()
    for i, story in enumerate(_STORIES):
        (src / f"story_{i}.txt").write_text(story, encoding="utf-8")
    return src


def test_prepare_data_from_local_files(tmp_path: Path, corpus_dir: Path) -> None:
    output = tmp_path / "corpus.jsonl"
    args = argparse.Namespace(
        config=_write_yaml(
            tmp_path / "prepare.yaml",
            {
                "source": {"type": "local_file", "root": str(corpus_dir)},
                "processing": [
                    {"type": "unicode_normalize"},
                    {"type": "quality_filter", "min_chars": 5},
                ],
            },
        ),
        output=str(output),
    )
    assert PrepareDataCommand().run(args) == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(_STORIES)
    assert all(json.loads(line)["text"] for line in lines)


def test_full_pipeline_prepare_tokenize_pack_train_generate(
    tmp_path: Path, corpus_dir: Path
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    PrepareDataCommand().run(
        argparse.Namespace(
            config=_write_yaml(
                tmp_path / "prepare.yaml",
                {
                    "source": {"type": "local_file", "root": str(corpus_dir)},
                    "processing": [{"type": "unicode_normalize"}],
                },
            ),
            output=str(corpus_path),
        )
    )

    tokenizer_path = tmp_path / "tokenizer.json"
    TrainTokenizerCommand().run(
        argparse.Namespace(
            config=_write_yaml(
                tmp_path / "tokenizer.yaml",
                {"tokenizer": {"type": "byte_level_bpe", "vocab_size": 300}},
            ),
            corpus=str(corpus_path),
            output=str(tokenizer_path),
        )
    )
    assert tokenizer_path.is_file()

    shard_dir = tmp_path / "shards"
    PackCommand().run(
        argparse.Namespace(
            corpus=str(corpus_path),
            tokenizer=str(tokenizer_path),
            output=str(shard_dir),
            tokens_per_shard=1000,
        )
    )
    assert (shard_dir / "manifest.json").is_file()

    checkpoint_dir = tmp_path / "checkpoints"
    train_config = _write_yaml(
        tmp_path / "train.yaml",
        {
            "model": {
                "vocab_size": 300,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
                "block_size": 8,
                "dropout": 0.0,
            },
            "trainer": {
                "max_steps": 3,
                "micro_batch_size": 2,
                "device": "cpu",
                "precision": "fp32",
            },
            "lr_schedule": {"type": "cosine", "max_lr": 1e-3, "total_steps": 3},
            "data": {"shard_dir": str(shard_dir)},
            "checkpoint_dir": str(checkpoint_dir),
            "log_every": 1,
            "checkpoint_every": 3,
        },
    )
    assert TrainCommand().run(argparse.Namespace(config=train_config)) == 0
    assert list(checkpoint_dir.glob("step_*.pt"))

    model_config_path = _write_yaml(
        tmp_path / "model_only.yaml",
        {
            "model": {
                "vocab_size": 300,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
                "block_size": 8,
                "dropout": 0.0,
            }
        },
    )
    generate_args = argparse.Namespace(
        model_config=model_config_path,
        checkpoint=str(checkpoint_dir),
        tokenizer=str(tokenizer_path),
        prompt="The",
        max_new_tokens=5,
        temperature=0.8,
        top_k=0,
        top_p=1.0,
        seed=0,
        device="cpu",
    )
    assert GenerateCommand().run(generate_args) == 0


def test_ablate_runs_every_config_in_directory(tmp_path: Path, corpus_dir: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    PrepareDataCommand().run(
        argparse.Namespace(
            config=_write_yaml(
                tmp_path / "prepare.yaml",
                {"source": {"type": "local_file", "root": str(corpus_dir)}, "processing": []},
            ),
            output=str(corpus_path),
        )
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    TrainTokenizerCommand().run(
        argparse.Namespace(
            config=_write_yaml(
                tmp_path / "tokenizer.yaml",
                {"tokenizer": {"type": "byte_level_bpe", "vocab_size": 300}},
            ),
            corpus=str(corpus_path),
            output=str(tokenizer_path),
        )
    )
    shard_dir = tmp_path / "shards"
    PackCommand().run(
        argparse.Namespace(
            corpus=str(corpus_path), tokenizer=str(tokenizer_path), output=str(shard_dir),
            tokens_per_shard=1000,
        )
    )

    config_dir = tmp_path / "ablation_configs"
    config_dir.mkdir()
    base_model = {
        "vocab_size": 300, "n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8, "dropout": 0.0,
    }
    for name, attention in [("s0", "vanilla_mha"), ("s1", "causal_self_attention")]:
        _write_yaml(
            config_dir / f"{name}.yaml",
            {
                "model": {**base_model, "attention": attention},
                "trainer": {
                    "max_steps": 2, "micro_batch_size": 2, "device": "cpu", "precision": "fp32"
                },
                "lr_schedule": {"type": "cosine", "max_lr": 1e-3, "total_steps": 2},
                "data": {"shard_dir": str(shard_dir)},
                "log_every": 1,
            },
        )

    summary_path = tmp_path / "summary.json"
    result = AblateCommand().run(
        argparse.Namespace(config_dir=str(config_dir), summary=str(summary_path))
    )
    assert result == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary) == {"s0", "s1"}
    assert all(entry["final_step"] == 2 for entry in summary.values())


def _write_yaml(path: Path, data: dict[str, object]) -> str:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)
