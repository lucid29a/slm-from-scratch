# Measured throughput on an RTX 4070 Laptop (8 GB), 151M non-embedding params

**Measured 2026-08-25**, replacing the earlier *estimated* 35–45 h figure in the
project plan, which was wrong by roughly 2.5×. Reproduce with:

```bash
python scripts/benchmark_throughput.py --micro-batch 4 --steps 8 --warmup-steps 2 --compile
```

Model: 24 layers, 768-dim, 12 heads / 4 KV heads, block 1024, RoPE + RMSNorm +
SwiGLU + GQA, weight-tied, bf16. 151,032,576 non-embedding parameters.

| Config | tokens/s | TFLOP/s | Peak VRAM | 1B tokens | 3B tokens |
|---|---|---|---|---|---|
| micro-batch 4, eager | 8,668 | 7.85 | 7.42 GB | 32 h | **96 h** |
| micro-batch 4, `torch.compile` | **12,430** | 11.26 | 5.53 GB | **22 h** | 67 h |
| micro-batch 8, `torch.compile` | 2,288 | 2.07 | 8.60 GB | 121 h | 364 h |

## Findings

**1. The original 35–45 h estimate was wrong.** The honest baseline for
3B tokens (Chinchilla-optimal 20 tokens/param) is **~96 hours — four days**, not
35–45. The earlier figure came from assuming an MFU we never measured.

**2. `torch.compile` is worth 1.43× and *saves* 1.9 GB.** 8,668 → 12,430
tokens/s, and peak VRAM drops from 7.42 GB to 5.53 GB because Inductor fuses
away intermediate activations. On Windows this needs `triton-windows`
(available as a Python 3.14 wheel); it is not installed by default with
PyTorch on this platform.

**3. 8 GB of VRAM is a hard cliff, not a gentle slope.** Doubling micro-batch
from 4 to 8 pushes peak VRAM to 8.60 GB — past the card's 8 GB — and
throughput *collapses* by 5.4× (12,430 → 2,288 tokens/s) as allocations spill
to host memory. This is the single most important operational fact for this
project's hardware: the usual "just raise the batch size for better
utilization" advice inverts here. Effective batch size must be grown with
gradient accumulation, never with micro-batch, once VRAM is near the limit.

This cliff also explains two earlier confusing observations: benchmark runs at
micro-batch 8 that appeared to hang were thrashing, not stuck.

## Implication for the planned 150M run

At the measured 12,430 tokens/s with `torch.compile`:

- 3B tokens (Chinchilla 20×/param): **67 h** ≈ 2.8 days
- 1B tokens (the single-GPU nanoGPT speedrun's budget for GPT-2-matching loss): **22 h**

The token budget, not the kernel speed, is the dominant term. See
`docs/speedup_plan.md` for the stacked-technique plan built on these numbers.
