"""``slm generate``: sample text from a trained checkpoint.

A self-contained, KV-cached sampling loop (greedy / temperature / top-k / top-p).
This will move to (and be reused by) ``slm_from_scratch.inference`` once that
subsystem exists; until then, the CLI owns it directly rather than blocking on
a phase that hasn't been built yet.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.cli.util import load_yaml_mapping
from slm_from_scratch.modeling import DecoderOnlyTransformer, ModelConfig
from slm_from_scratch.tokenization.base import Tokenizer
from slm_from_scratch.training.checkpoint import CheckpointManager

__all__ = ["GenerateCommand"]


@COMMANDS.register("generate")
class GenerateCommand(Command):
    """Load a checkpoint and generate a continuation of a prompt."""

    name = "generate"
    help = "Generate text from a trained checkpoint."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register model/checkpoint/tokenizer/sampling arguments."""
        parser.add_argument(
            "--model-config", required=True, help="YAML file with a `model:` section."
        )
        parser.add_argument(
            "--checkpoint", required=True, help="Checkpoint directory or a .pt file."
        )
        parser.add_argument("--tokenizer", required=True, help="Path to a saved tokenizer.")
        parser.add_argument("--prompt", required=True, help="Text prompt to continue.")
        parser.add_argument("--max-new-tokens", type=int, default=200)
        parser.add_argument("--temperature", type=float, default=0.8)
        parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering.")
        parser.add_argument(
            "--top-p", type=float, default=1.0, help="1.0 disables nucleus filtering."
        )
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    def run(self, args: argparse.Namespace) -> int:
        """Load the model + tokenizer and print a generated continuation."""
        if args.seed is not None:
            torch.manual_seed(args.seed)

        spec = load_yaml_mapping(args.model_config)
        model_config = ModelConfig.from_dict(spec["model"])
        model = DecoderOnlyTransformer(model_config)

        checkpoint_manager = CheckpointManager(args.checkpoint)
        state = checkpoint_manager.load_latest()
        if state is None:
            raise SystemExit(f"no checkpoint found in {args.checkpoint}")
        model.load_state_dict(state.model_state)

        device = torch.device(args.device)
        model = model.to(device)
        model.eval()

        tokenizer = Tokenizer.load(args.tokenizer)

        text = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            top_p=args.top_p,
            device=device,
        )
        print(text)
        return 0


@torch.no_grad()
def generate(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float = 1.0,
    device: torch.device,
) -> str:
    """Sample a continuation of ``prompt`` using a KV cache.

    Args:
        model: A trained :class:`DecoderOnlyTransformer`, already on ``device``.
        tokenizer: The tokenizer the model was trained with.
        prompt: Text to continue from.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature; ``0`` degenerates to greedy decoding.
        top_k: Restrict sampling to the ``top_k`` most likely tokens, or ``None``
            to disable.
        top_p: Nucleus sampling threshold; ``1.0`` disables it.
        device: Device to run generation on.

    Returns:
        The decoded prompt + generated continuation.
    """
    ids = tokenizer.encode(prompt)
    generated = list(ids)
    kv_cache = model.new_kv_cache()

    current = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(current, kv_cache=kv_cache)
        next_logits = logits[0, -1]
        next_id = _sample(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        current = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return str(tokenizer.decode(generated))


def _sample(
    logits: torch.Tensor, *, temperature: float, top_k: int | None, top_p: float
) -> int:
    if temperature <= 0:
        return int(logits.argmax().item())

    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.size(-1):
        threshold = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())
