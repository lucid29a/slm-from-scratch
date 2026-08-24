# Phase 6: the ablation ladder — results, and an honest surprise

**Run 2026-08-24.** Reproduce with:

```bash
slm ablate --config-dir configs/ablation --summary artifacts/ablation/summary.json
# then, per rung:
slm eval --config configs/eval/s0_vanilla.yaml   # ... s1 .. s6
```

## Setup

Seven ~50M-non-embedding-parameter models, each changing exactly one
architectural choice from the previous (see [`configs/ablation/`](../configs/ablation/)):

S0 vanilla 2017 (post-norm, learned PE, GELU, plain MHA, LayerNorm, untied
embeddings) → S1 pre-norm → S2 RMSNorm → S3 RoPE → S4 SwiGLU → S5 GQA → S6
weight tying. All seven share the same seed, batch size, and step count
(6,000 steps, batch 32 × block 256 = ~49.2M training tokens sampled with
replacement), trained on the Phase 5 TinyStories shards (~4.57M unique
tokens — i.e. roughly **11 effective epochs** over the same data).

Held-out evaluation used a *disjoint* 5,000-document TinyStories slice
(documents 20,000–25,000, never seen during training), packed with the same
tokenizer into its own shards.

## Results

| Rung | Train loss | Held-out perplexity | Held-out bits/byte | Held-out token acc. |
|---|---|---|---|---|
| S0 vanilla | 1.416 | **8.28** | **0.735** | **0.529** |
| S1 pre-norm | 1.169 | 9.47 | 0.782 | 0.519 |
| S2 RMSNorm | 1.160 | 9.85 | 0.796 | 0.513 |
| S3 RoPE | 0.984 | 11.87 | 0.861 | 0.498 |
| S4 SwiGLU | 0.666 | 15.59 | 0.955 | 0.494 |
| S5 GQA | 0.675 | 16.78 | 0.981 | 0.490 |
| S6 full modern | 0.633 | 16.16 | 0.968 | 0.481 |

## The honest finding: the two rankings invert

**By training loss, S6 (full modern stack) wins decisively** — 55% lower
loss than S0, matching every prior in the SLM literature about RoPE/SwiGLU
being strict upgrades. **By held-out perplexity, S0 (the 2017 vanilla
baseline) wins just as decisively** — it generalizes best, and every
subsequent "upgrade" makes held-out perplexity *worse*, monotonically.

This is not a bug in the evaluation code (`bits_per_byte` and
`token_accuracy` — measured completely independently of `perplexity` — tell
the identical story, and the held-out set is genuinely disjoint from
training). It's **overfitting, and it's a direct, predictable consequence of
this run's token budget**: 6,000 steps × 8,192 tokens/step ≈ 49M training
tokens drawn *with replacement* from a **4.57M-token** pool is about 11
epochs over the same text. SwiGLU and RoPE are more expressive / converge
faster — which under a data-starved budget means they memorize the training
set harder, not that they generalize worse in general. The training-loss
table (what `slm ablate`'s summary reports on its own) would have told a
paper-friendly but misleading story; only pairing it with a genuine held-out
set catches this.

**This is exactly why held-out evaluation was built before the 150M run**, not
after: the 150M run is planned at Chinchilla-optimal ≈20 tokens/param, i.e.
~3B *unique* tokens — enough that this specific failure mode (many epochs
over a tiny corpus) shouldn't recur. But it means this ablation ladder's
*ranking* isn't reusable as the paper's evidence for which architecture to
scale up; at minimum it needs to be rerun at a token budget with enough
unique data that the runs are compute-bound, not epoch-bound (e.g. a single
epoch, or capped at 2-3 epochs, over a corpus sized so 6,000 steps stays
within it).

## What's still valid

- **The infrastructure is validated end-to-end**: `slm ablate` correctly ran seven
  independent configs, correct incremental diffs (tested), correct param
  counts through GQA/weight-tying, and `slm eval` produced numbers on
  genuinely held-out data — all the machinery Phase 7 needs is proven, live,
  not just unit-tested.
- **The overfitting signature itself is a legitimate, citable finding** for
  the paper's methodology section: it's a concrete demonstration of why token
  budget, not just architecture, determines whether "modern" components help.

## Next step

Before trusting an architecture ranking for the paper, rerun the ladder with
either (a) a larger unique-token pool (e.g. a FineWeb-Edu slice sized to keep
each rung under ~1.5 epochs), or (b) fewer steps so the existing TinyStories
pool isn't over-visited. Given Phase 7's 150M run already requires ~3B unique
tokens, the more efficient path is likely to pack that corpus once and reuse
a slice of it for a corrected ablation rerun, rather than sourcing new data
twice.
