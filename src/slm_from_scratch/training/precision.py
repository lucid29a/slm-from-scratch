"""Mixed-precision policy: a thin, testable wrapper around ``torch.autocast``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from slm_from_scratch.core.exceptions import ConfigError

__all__ = ["PrecisionPolicy"]

_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class PrecisionPolicy:
    """Chooses a compute dtype and wraps the forward pass in ``torch.autocast``.

    Args:
        precision: One of ``"fp32"``, ``"fp16"``, ``"bf16"``.
        device_type: The device kind autocast should target (``"cuda"`` or ``"cpu"``).

    Raises:
        ConfigError: If ``precision`` isn't one of the three supported values.
    """

    def __init__(self, precision: str = "bf16", *, device_type: str = "cuda") -> None:
        if precision not in _DTYPES:
            raise ConfigError(f"unknown precision {precision!r}; choose from {list(_DTYPES)}")
        self.precision = precision
        self.device_type = device_type
        self.dtype = _DTYPES[precision]
        self.enabled = precision != "fp32"

    @contextmanager
    def autocast(self) -> Iterator[None]:
        """Context manager: runs enclosed ops in this policy's dtype (no-op for fp32)."""
        with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enabled):
            yield
