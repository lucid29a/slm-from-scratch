# Phase 5 gate: the 5-minute TinyStories smoke test

**Status: passed, 2026-08-24.** Reproduce with:

```bash
python scripts/run_tinystories_smoke.py
```

## Setup

- 20,000 TinyStories documents (streamed from `roneneldan/TinyStories`)
- Byte-level BPE tokenizer trained from scratch, vocab size 4,096
- ~4.57M tokens packed to binary shards, block size 256
- Model: 6 layers, 8 heads, 256-dim, RoPE + RMSNorm + SwiGLU + fused causal
  attention, weight-tied -- **4,718,848 non-embedding parameters**
- 1,500 steps, batch size 32, bf16, cosine LR schedule (peak 3e-4), on one
  RTX 4070 Laptop GPU (8 GB)

## Result

| Metric | Value |
|---|---|
| Total wall time (download + tokenize + pack + train) | **2.01 min** |
| Training-only wall time | 75.2 s |
| Loss: step 0 -> step 1400 | 8.32 -> 2.57 |

Final sample (`"Once upon a time"`, greedy decoding):

> Once upon a time, there was a little girl named Lily. She loved to play
> outside in the sky. One day, she saw a big, red ball. She wanted to play
> with it, but it was too high.
>
> Lily asked her mom, "Can I play with it?" Her mom said, "Yes, we can play
> with it."
>
> Lily was happy to see the ball and they played with it together. They
> played with it all day long. The ball was happy to have a new friend.

Coherent, grammatical, on-topic, and the model chose to end its own story
with `<eos>` rather than running out the generation budget. This is the
plan's definition-of-done for Phase 5: a reader can go from a clean checkout
to a trained, readable model in about two minutes, without a GPU cluster.
