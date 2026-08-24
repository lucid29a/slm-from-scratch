"""``slm train``: build a model + Trainer from one YAML config and run it.

Config file shape::

    model:
      vocab_size: 4096
      n_layer: 6
      n_head: 8
      n_embd: 256
      block_size: 256
      attention: causal_self_attention
      positional_encoding: rotary
      normalization: rmsnorm
      feedforward: swiglu

    trainer:
      max_steps: 1500
      micro_batch_size: 32
      precision: bf16
      device: cuda

    optimizer:
      learning_rate: 3.0e-4

    lr_schedule:
      type: cosine
      max_lr: 3.0e-4
      min_lr: 3.0e-5
      warmup_steps: 100
      total_steps: 1500

    data:
      shard_dir: artifacts/shards

    checkpoint_dir: artifacts/checkpoints
    log_every: 100
    checkpoint_every: 500

    # optional: generate a sample every N steps during training
    tokenizer: artifacts/tokenizer.json
    sample_every: 500
    sample_prompt: "Once upon a time"
"""

from __future__ import annotations

import argparse
from typing import Any

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.cli.util import load_yaml_mapping
from slm_from_scratch.data.packing import MemmapTokenDataset
from slm_from_scratch.modeling import DecoderOnlyTransformer, ModelConfig
from slm_from_scratch.tokenization.base import Tokenizer
from slm_from_scratch.training import (
    LR_SCHEDULES,
    Callback,
    CheckpointCallback,
    CheckpointManager,
    ConsoleLogger,
    LoggingCallback,
    LRScheduleConfig,
    OptimizerConfig,
    SampleGenerationCallback,
    Trainer,
    TrainerConfig,
)

__all__ = ["TrainCommand", "build_trainer"]


def build_trainer(spec: dict[str, Any]) -> Trainer:
    """Build a fully wired :class:`Trainer` from a parsed training-config mapping.

    Shared by :class:`TrainCommand` and :class:`~slm_from_scratch.cli.commands.ablate.AblateCommand`
    so both build a run identically from a YAML mapping of the shape documented
    in this module's docstring.
    """
    model_config = ModelConfig.from_dict(spec["model"])
    model = DecoderOnlyTransformer(model_config)
    print(f"train: model has {model.num_parameters():,} non-embedding parameters")

    dataset = MemmapTokenDataset(spec["data"]["shard_dir"], block_size=model_config.block_size)
    print(f"train: {len(dataset):,} training positions")

    trainer_config = TrainerConfig.from_dict(spec["trainer"])
    optimizer_config = OptimizerConfig.from_dict(spec.get("optimizer") or {})

    schedule_spec = dict(spec["lr_schedule"])
    schedule_kind = schedule_spec.pop("type")
    schedule_cls = LR_SCHEDULES.get(schedule_kind)
    lr_schedule = schedule_cls(LRScheduleConfig.from_dict(schedule_spec))

    log_every = spec.get("log_every", 10)
    callbacks: list[Callback] = [LoggingCallback(ConsoleLogger(), log_every=log_every)]

    checkpoint_manager = None
    if spec.get("checkpoint_dir"):
        checkpoint_manager = CheckpointManager(spec["checkpoint_dir"])
        callbacks.append(
            CheckpointCallback(checkpoint_manager, every_n_steps=spec.get("checkpoint_every", 500))
        )

    if spec.get("tokenizer") and spec.get("sample_every"):
        tokenizer = Tokenizer.load(spec["tokenizer"])
        callbacks.append(
            SampleGenerationCallback(
                tokenizer,
                prompt=spec.get("sample_prompt", "Once upon a time"),
                every_n_steps=spec["sample_every"],
            )
        )

    return Trainer(
        model=model,
        train_dataset=dataset,
        config=trainer_config,
        optimizer_config=optimizer_config,
        lr_schedule=lr_schedule,
        callbacks=callbacks,
        checkpoint_manager=checkpoint_manager,
    )


@COMMANDS.register("train")
class TrainCommand(Command):
    """Train a model from a single YAML config.

    Resumes automatically if a checkpoint already exists in ``checkpoint_dir``.
    """

    name = "train"
    help = "Train a model from a YAML config (model + trainer + optimizer + lr_schedule + data)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register ``--config``."""
        parser.add_argument("--config", required=True, help="Path to a train YAML config.")

    def run(self, args: argparse.Namespace) -> int:
        """Build model/dataset/trainer from ``--config`` and run training to completion."""
        spec = load_yaml_mapping(args.config)
        trainer = build_trainer(spec)
        if trainer.current_step > 0:
            print(f"train: resumed from step {trainer.current_step}")

        trainer.train()
        print(f"train: finished at step {trainer.current_step}")
        return 0
