"""Core abstractions shared by every subsystem: registry, config, component, protocols."""

from __future__ import annotations

from slm_from_scratch.core.component import Component
from slm_from_scratch.core.config import BaseConfig, load_yaml_with_extends
from slm_from_scratch.core.exceptions import (
    CheckpointError,
    ConfigError,
    DataError,
    EvaluationError,
    ModelError,
    RegistryError,
    SLMError,
    TokenizerError,
)
from slm_from_scratch.core.registry import Registry

__all__ = [
    "BaseConfig",
    "CheckpointError",
    "Component",
    "ConfigError",
    "DataError",
    "EvaluationError",
    "ModelError",
    "Registry",
    "RegistryError",
    "SLMError",
    "TokenizerError",
    "load_yaml_with_extends",
]
