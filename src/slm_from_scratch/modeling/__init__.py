"""The model: strategies composed into transformer blocks and a full model.

Attention, positional encoding, normalization, and feedforward strategies,
composed into transformer blocks and a decoder-only transformer. Importing
this package registers every concrete strategy in its registry
(``ATTENTION``, ``POSITIONAL_ENCODING``, ``NORMALIZATION``, ``FEEDFORWARD``,
``INITIALIZERS``, ``MODELS``).
"""

from __future__ import annotations

from slm_from_scratch.modeling.attention import (
    ATTENTION,
    AttentionBase,
    CausalSelfAttention,
    GroupedQueryAttention,
    VanillaMultiHeadAttention,
)
from slm_from_scratch.modeling.base import MODELS, LanguageModel, ModelConfig
from slm_from_scratch.modeling.block import TransformerBlock
from slm_from_scratch.modeling.feedforward import (
    FEEDFORWARD,
    FeedForwardBase,
    GELUFeedForward,
    SwiGLUFeedForward,
)
from slm_from_scratch.modeling.init import INITIALIZERS, NormalWeightInit, WeightInitializer
from slm_from_scratch.modeling.kv_cache import KVCache, LayerKVCache
from slm_from_scratch.modeling.normalization import (
    NORMALIZATION,
    LayerNorm,
    NormalizationBase,
    RMSNorm,
)
from slm_from_scratch.modeling.positional import (
    POSITIONAL_ENCODING,
    ALiBi,
    LearnedPositionalEmbedding,
    PositionalEncodingBase,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from slm_from_scratch.modeling.transformer import DecoderOnlyTransformer

__all__ = [
    "ATTENTION",
    "FEEDFORWARD",
    "INITIALIZERS",
    "MODELS",
    "NORMALIZATION",
    "POSITIONAL_ENCODING",
    "ALiBi",
    "AttentionBase",
    "CausalSelfAttention",
    "DecoderOnlyTransformer",
    "FeedForwardBase",
    "GELUFeedForward",
    "GroupedQueryAttention",
    "KVCache",
    "LanguageModel",
    "LayerKVCache",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "ModelConfig",
    "NormalWeightInit",
    "NormalizationBase",
    "PositionalEncodingBase",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SinusoidalPositionalEncoding",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "VanillaMultiHeadAttention",
    "WeightInitializer",
]
