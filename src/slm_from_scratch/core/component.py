"""The common base for every buildable, config-carrying object in the project."""

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

from slm_from_scratch.core.config import BaseConfig

__all__ = ["Component"]

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class Component(ABC, Generic[ConfigT]):
    """Base class for objects that are built from, and remember, a config.

    A :class:`Component` is the unit the registries in this project traffic in:
    tokenizers, attention modules, callbacks, text sources, samplers. Each one
    keeps a reference to the config it was built from, so the object can always
    answer "what were you configured to do" -- useful for logging, checkpoint
    metadata, and reproducing a run.

    Args:
        config: The validated configuration this component was built from.
    """

    def __init__(self, config: ConfigT) -> None:
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this component was constructed from."""
        return self._config

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._config!r})"
