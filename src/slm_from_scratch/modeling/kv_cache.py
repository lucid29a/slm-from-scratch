"""The KV cache: first-class objects, not tensor tuples threaded through every call.

During autoregressive generation, each new token only needs to attend to the
keys/values of *previous* tokens -- recomputing them every step is wasted work.
A :class:`KVCache` holds one :class:`LayerKVCache` per transformer layer, each
of which appends newly computed keys/values to what it already has and hands
back the full history for that layer's attention to use.
"""

from __future__ import annotations

import torch

__all__ = ["KVCache", "LayerKVCache"]


class LayerKVCache:
    """Accumulates one transformer layer's key/value tensors across decode steps."""

    def __init__(self) -> None:
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full history so far.

        Args:
            k: New keys, ``(batch, n_kv_head, new_seq_len, head_dim)``.
            v: New values, same shape as ``k``.

        Returns:
            ``(k, v)`` concatenated with anything previously cached, along the
            sequence dimension.
        """
        if self._k is None:
            self._k, self._v = k, v
        else:
            assert self._v is not None
            self._k = torch.cat([self._k, k], dim=2)
            self._v = torch.cat([self._v, v], dim=2)
        return self._k, self._v

    @property
    def seq_len(self) -> int:
        """Number of positions currently cached (0 if empty)."""
        return 0 if self._k is None else self._k.size(2)

    def reset(self) -> None:
        """Discard all cached state."""
        self._k = None
        self._v = None


class KVCache:
    """One :class:`LayerKVCache` per transformer layer."""

    def __init__(self, n_layer: int) -> None:
        self._layers = [LayerKVCache() for _ in range(n_layer)]

    def layer(self, index: int) -> LayerKVCache:
        """Return the cache for transformer layer ``index``."""
        return self._layers[index]

    @property
    def seq_len(self) -> int:
        """Number of positions cached so far (same across all layers by construction)."""
        return self._layers[0].seq_len if self._layers else 0

    def reset(self) -> None:
        """Discard all cached state in every layer."""
        for layer in self._layers:
            layer.reset()

    def __len__(self) -> int:
        return len(self._layers)
