"""Measure real training throughput for a given model config, to replace guesswork
in wall-clock estimates.

Reports tokens/sec, achieved TFLOPs, and MFU using the standard 6*N*D
approximation, optionally with ``torch.compile`` enabled so its speedup can be
measured rather than assumed.

Usage:
    python scripts/benchmark_throughput.py --n-layer 24 --n-embd 768 --block-size 1024 \
        --micro-batch 8 --grad-accum 1 --steps 30 --compile
"""

from __future__ import annotations

import argparse
import time

import torch

from slm_from_scratch.modeling import DecoderOnlyTransformer, ModelConfig
from slm_from_scratch.training.gradient import GradientClipper
from slm_from_scratch.training.optimizer import OptimizerConfig, OptimizerFactory
from slm_from_scratch.training.precision import PrecisionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-kv-head", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    parser.add_argument(
        "--activation-checkpointing", action="store_true", help="Trade compute for memory."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    config = ModelConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        n_kv_head=args.n_kv_head,
        dropout=0.0,
        attention="grouped_query_attention",
        positional_encoding="rotary",
        normalization="rmsnorm",
        norm_placement="pre",
        feedforward="swiglu",
        weight_tying=True,
        bias=False,
    )
    model = DecoderOnlyTransformer(config).to(device)
    num_params = model.num_parameters()
    print(f"non-embedding params: {num_params:,}")

    if args.compile:
        print("compiling (first step will be slow) ...")
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = OptimizerFactory(OptimizerConfig(learning_rate=3e-4)).build(model)
    precision = PrecisionPolicy(args.precision, device_type=device.type)
    clipper = GradientClipper(1.0)

    tokens_per_step = args.micro_batch * args.grad_accum * args.block_size

    def one_step() -> None:
        for _ in range(args.grad_accum):
            inputs = torch.randint(
                0, args.vocab_size, (args.micro_batch, args.block_size), device=device
            )
            targets = torch.randint(
                0, args.vocab_size, (args.micro_batch, args.block_size), device=device
            )
            with precision.autocast():
                _, loss = model(inputs, targets)
            (loss / args.grad_accum).backward()
        clipper.clip(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    steps_per_sec = args.steps / elapsed
    tokens_per_sec = steps_per_sec * tokens_per_step
    achieved_flops = 6 * num_params * tokens_per_sec

    print(f"\n--- {args.steps} steps in {elapsed:.2f}s ---")
    print(f"steps/sec:    {steps_per_sec:.3f}")
    print(f"tokens/sec:   {tokens_per_sec:,.0f}")
    print(f"achieved:     {achieved_flops / 1e12:.2f} TFLOP/s (6*N*D approximation)")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM:    {peak_gb:.2f} GB")

    for budget_name, budget in [("1B tokens", 1e9), ("3B tokens (Chinchilla 20x)", 3e9)]:
        hours = budget / tokens_per_sec / 3600
        print(f"{budget_name:<28} -> {hours:6.1f} h ({hours / 24:.1f} days)")


if __name__ == "__main__":
    main()
