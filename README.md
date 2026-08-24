# slm-from-scratch

> Build a Small Language Model from zero — a rigorously object-oriented codebase
> that doubles as a curriculum. Tokenizer, data pipeline, transformer, trainer,
> evaluator, and inference stack, all implemented by hand, all swappable through
> config.

**Status: early scaffolding (Phase 1 of the build).** Architecture, docs, and the
paper are in progress — see [`docs/`](docs/) as chapters land and
[`paper/`](paper/) for the accompanying position paper and technical report.

## Why this exists

Most "build a GPT" repos are single-file scripts (great for reading, useless as a
foundation) or full frameworks (great as a foundation, opaque as a lesson). This
project tries to be both: every architectural choice — attention variant,
normalization, positional encoding, feed-forward block — is an abstract base class
with concrete, independently swappable implementations, registered under a short
config key. That is also what makes the paper's ablation study possible: swapping
`normalization: layernorm` for `normalization: rmsnorm` in a YAML file *is* the
diff between two experiments.

## The ladder

| Scale | Purpose | Target hardware |
|---|---|---|
| ~5M params | Read the whole pipeline run end to end in minutes | Any recent GPU, even a laptop |
| ~50M params | The ablation study (vanilla transformer → modern) | A single 8 GB GPU, hours |
| ~150M params | The real trained model behind the paper's numbers | A single 8 GB GPU, ~1–3 days |

## Project layout

```
src/slm_from_scratch/   the library: core, tokenization, data, modeling,
                         training, evaluation, inference, cli
configs/                 YAML configs for models, data, training, ablations
docs/                     the book — one chapter per subsystem
notebooks/                worked examples
paper/                    LaTeX sources for the position paper / technical report
tests/                    unit + integration tests, mirrors src/
```

## Quickstart

Coming online as each phase lands. The eventual first command:

```bash
pip install -e ".[dev,data]"
slm train --config configs/train/tiny.yaml
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
