"""The decoder-only transformer, assembled from the strategies named in a config.

Token embedding, a stack of blocks, and an output head, built entirely from the
registered strategies named in a
:class:`~slm_from_scratch.modeling.base.ModelConfig`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from slm_from_scratch.modeling.base import MODELS, LanguageModel, ModelConfig
from slm_from_scratch.modeling.block import TransformerBlock
from slm_from_scratch.modeling.init import NormalWeightInit, WeightInitializer
from slm_from_scratch.modeling.kv_cache import KVCache
from slm_from_scratch.modeling.normalization import NORMALIZATION
from slm_from_scratch.modeling.positional import POSITIONAL_ENCODING

__all__ = ["DecoderOnlyTransformer"]


@MODELS.register("decoder_only_transformer")
class DecoderOnlyTransformer(LanguageModel):
    """A GPT-style decoder-only transformer, built from :data:`ModelConfig`'s strategies.

    Args:
        config: The model configuration.
        initializer: Weight-initialization strategy; defaults to
            :class:`~slm_from_scratch.modeling.init.NormalWeightInit`.
    """

    def __init__(
        self, config: ModelConfig, *, initializer: WeightInitializer | None = None
    ) -> None:
        super().__init__(config)

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional = POSITIONAL_ENCODING.build(
            config.positional_encoding,
            n_embd=config.n_embd,
            n_head=config.n_head,
            head_dim=config.head_dim,
            block_size=config.block_size,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = NORMALIZATION.build(config.normalization, config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        init_strategy = initializer if initializer is not None else NormalWeightInit()
        init_strategy.initialize(self, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the transformer, optionally against a KV cache for incremental decoding.

        Args:
            input_ids: ``(batch, seq_len)`` token ids.
            targets: Optional ``(batch, seq_len)`` next-token ids for computing loss.
            kv_cache: Optional cache for incremental (one-token-at-a-time) decoding;
                its current length determines each new token's absolute position.

        Returns:
            ``(logits, loss)`` -- ``logits`` is ``(batch, seq_len, vocab_size)``;
            ``loss`` is ``None`` unless ``targets`` was given.
        """
        config = self.config
        assert isinstance(config, ModelConfig)
        _, seq_len = input_ids.shape
        if seq_len > config.block_size and kv_cache is None:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {config.block_size}"
            )

        start_pos = kv_cache.seq_len if kv_cache is not None else 0

        x = self.token_embedding(input_ids)
        x = self.positional.encode_embeddings(x, start_pos=start_pos)
        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache.layer(i) if kv_cache is not None else None
            x = block(x, positional=self.positional, start_pos=start_pos, kv_cache=layer_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def num_parameters(self, *, non_embedding: bool = True) -> int:
        """Count trainable parameters, optionally excluding the token embedding table."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
        return total

    def new_kv_cache(self) -> KVCache:
        """Create an empty :class:`KVCache` sized for this model's layer count."""
        config = self.config
        assert isinstance(config, ModelConfig)
        return KVCache(config.n_layer)
