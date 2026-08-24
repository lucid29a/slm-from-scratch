"""Tests for TransformerBlock."""

from __future__ import annotations

import pytest
import torch

from slm_from_scratch.modeling.block import TransformerBlock
from tests.modeling.conftest import make_config


@pytest.mark.parametrize("norm_placement", ["pre", "post"])
def test_output_shape_matches_input(norm_placement: str) -> None:
    config = make_config(norm_placement=norm_placement)
    block = TransformerBlock(config)
    x = torch.randn(2, 6, config.n_embd)
    from slm_from_scratch.modeling.positional import POSITIONAL_ENCODING

    positional = POSITIONAL_ENCODING.build(
        config.positional_encoding,
        n_embd=config.n_embd,
        n_head=config.n_head,
        head_dim=config.head_dim,
        block_size=config.block_size,
    )
    out = block(x, positional=positional)
    assert out.shape == x.shape


def test_gradients_flow_through_residual_connections() -> None:
    config = make_config()
    block = TransformerBlock(config)
    from slm_from_scratch.modeling.positional import POSITIONAL_ENCODING

    positional = POSITIONAL_ENCODING.build(
        config.positional_encoding,
        n_embd=config.n_embd,
        n_head=config.n_head,
        head_dim=config.head_dim,
        block_size=config.block_size,
    )
    x = torch.randn(2, 6, config.n_embd, requires_grad=True)
    block(x, positional=positional).sum().backward()
    assert x.grad is not None
    # Residual connections guarantee no vanishing gradient at the input even
    # through an (untrained, randomly-initialized) attention + FFN stack.
    assert x.grad.abs().sum().item() > 0
