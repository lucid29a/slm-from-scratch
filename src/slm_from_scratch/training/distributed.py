"""Distributed-training strategy: single-device today, DDP as an honest stub.

This project was built and validated on a single laptop GPU. :class:`DDPStrategy`
exists so the abstraction is complete and the training loop never special-cases
"am I distributed?" -- but it has not been run or tested against multiple
devices, and is documented here as such rather than silently implied to work.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

__all__ = ["DDPStrategy", "DistributedStrategy", "SingleDeviceStrategy"]


class DistributedStrategy(ABC):
    """Abstract base for how the training loop talks to one or many devices."""

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap ``model`` for this strategy's execution mode (e.g. in DDP)."""
        raise NotImplementedError

    @abstractmethod
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return the underlying (un-wrapped) model, for state_dict/checkpointing."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        """Whether this process should do logging/checkpointing/eval."""
        raise NotImplementedError

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Total number of participating processes."""
        raise NotImplementedError

    @abstractmethod
    def barrier(self) -> None:
        """Block until every process reaches this point (no-op if single-process)."""
        raise NotImplementedError


class SingleDeviceStrategy(DistributedStrategy):
    """The default: one process, one device, no synchronization needed."""

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Return ``model`` unchanged."""
        return model

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return ``model`` unchanged."""
        return model

    @property
    def is_main_process(self) -> bool:
        """Always ``True``: the only process is the main one."""
        return True

    @property
    def world_size(self) -> int:
        """Always ``1``."""
        return 1

    def barrier(self) -> None:
        """No-op: nothing to synchronize with."""


class DDPStrategy(DistributedStrategy):
    """Multi-GPU training via ``torch.nn.parallel.DistributedDataParallel``.

    Expects the standard ``torchrun``-style environment variables (``RANK``,
    ``LOCAL_RANK``, ``WORLD_SIZE``) and an already-initialized process group.

    Warning:
        Not exercised in this project's test suite or training runs -- there
        was only ever one GPU available. Included for architectural
        completeness; treat it as unvalidated until it has actually been run.
    """

    def __init__(self) -> None:
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "DDPStrategy requires an initialized torch.distributed process "
                "group; call torch.distributed.init_process_group() first"
            )
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._world_size = torch.distributed.get_world_size()

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap ``model`` in ``DistributedDataParallel`` on this process's local device."""
        model = model.to(f"cuda:{self._local_rank}")
        return DistributedDataParallel(model, device_ids=[self._local_rank])

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return ``model.module`` if wrapped, else ``model`` unchanged."""
        return model.module if isinstance(model, DistributedDataParallel) else model

    @property
    def is_main_process(self) -> bool:
        """Whether this process has global rank 0."""
        return torch.distributed.get_rank() == 0

    @property
    def world_size(self) -> int:
        """Total number of processes in the job."""
        return self._world_size

    def barrier(self) -> None:
        """Block until every process in the group reaches this point."""
        torch.distributed.barrier()
