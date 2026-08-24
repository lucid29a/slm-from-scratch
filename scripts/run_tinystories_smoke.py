"""Phase 5 smoke gate: train the ~5M-param model on TinyStories in minutes.

Downloads a small slice of TinyStories, trains a byte-level BPE tokenizer,
packs it to binary shards, trains a small decoder-only transformer for a
fixed step budget, and generates a sample -- end to end, on a single GPU, in
well under the project's "readable text in ~5 minutes" gate.

Usage:
    python scripts/run_tinystories_smoke.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from slm_from_scratch.data import BinaryShardWriter, MemmapTokenDataset, SequencePacker
from slm_from_scratch.data.sources import TinyStoriesSource, TinyStoriesSourceConfig
from slm_from_scratch.modeling import DecoderOnlyTransformer, ModelConfig
from slm_from_scratch.tokenization import ByteLevelBPETokenizer, ByteLevelBPETokenizerConfig
from slm_from_scratch.training import (
    ConsoleLogger,
    CosineWithWarmup,
    LoggingCallback,
    LRScheduleConfig,
    OptimizerConfig,
    SampleGenerationCallback,
    Trainer,
    TrainerConfig,
)

ARTIFACT_DIR = Path("artifacts/smoke")
NUM_DOCS = 20_000
VOCAB_SIZE = 4096
BLOCK_SIZE = 256


def main() -> None:
    t_start = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] downloading {NUM_DOCS} TinyStories documents ...")
    source = TinyStoriesSource(TinyStoriesSourceConfig(limit=NUM_DOCS))
    docs = list(source)
    print(f"      got {len(docs)} documents in {time.perf_counter() - t_start:.1f}s")

    tokenizer_path = ARTIFACT_DIR / "tokenizer.json"
    print(f"[2/5] training a byte-level BPE tokenizer (vocab_size={VOCAB_SIZE}) ...")
    t0 = time.perf_counter()
    tokenizer = ByteLevelBPETokenizer(ByteLevelBPETokenizerConfig(vocab_size=VOCAB_SIZE))
    tokenizer.train(docs)
    tokenizer.save(tokenizer_path)
    print(f"      vocab_size={tokenizer.vocab_size} in {time.perf_counter() - t0:.1f}s")

    shard_dir = ARTIFACT_DIR / "shards"
    print("[3/5] packing to binary shards ...")
    t0 = time.perf_counter()
    packer = SequencePacker(tokenizer)
    token_stream = packer.pack(docs)
    writer = BinaryShardWriter(
        shard_dir, vocab_size=tokenizer.vocab_size, tokens_per_shard=20_000_000
    )
    manifest = writer.write(token_stream)
    print(f"      {manifest.total_tokens:,} tokens in {time.perf_counter() - t0:.1f}s")

    dataset = MemmapTokenDataset(shard_dir, block_size=BLOCK_SIZE)
    print(f"      {len(dataset):,} training positions at block_size={BLOCK_SIZE}")

    print("[4/5] training the ~5M-param model ...")
    t0 = time.perf_counter()
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=6,
        n_head=8,
        n_embd=256,
        block_size=BLOCK_SIZE,
        dropout=0.0,
        attention="causal_self_attention",
        positional_encoding="rotary",
        normalization="rmsnorm",
        norm_placement="pre",
        feedforward="swiglu",
        weight_tying=True,
        bias=False,
    )
    model = DecoderOnlyTransformer(model_config)
    non_embedding_params = model.num_parameters(non_embedding=True)
    print(f"      non-embedding params: {non_embedding_params:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = 1500
    trainer_config = TrainerConfig(
        max_steps=max_steps,
        micro_batch_size=32,
        grad_accum_steps=1,
        grad_clip=1.0,
        precision="bf16" if device == "cuda" else "fp32",
        seed=1337,
        device=device,
    )
    optimizer_config = OptimizerConfig(learning_rate=3e-4, weight_decay=0.1)
    schedule = CosineWithWarmup(
        LRScheduleConfig(max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=max_steps)
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        config=trainer_config,
        optimizer_config=optimizer_config,
        lr_schedule=schedule,
        callbacks=[
            LoggingCallback(ConsoleLogger(), log_every=100),
            SampleGenerationCallback(
                tokenizer, prompt="Once upon a time", max_new_tokens=80, every_n_steps=500
            ),
        ],
    )
    trainer.train()
    train_elapsed = time.perf_counter() - t0
    print(f"      trained {max_steps} steps on {device} in {train_elapsed:.1f}s")

    print("[5/5] final sample:")
    sampler = SampleGenerationCallback(
        tokenizer, prompt="Once upon a time", max_new_tokens=120, every_n_steps=1
    )
    print(sampler.generate(trainer))

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
