"""Exception hierarchy for the whole project.

Every error raised deliberately by this codebase derives from :class:`SLMError`, so
callers can distinguish "the library said no" from "something else broke".
"""

from __future__ import annotations

__all__ = [
    "CheckpointError",
    "ConfigError",
    "DataError",
    "EvaluationError",
    "ModelError",
    "RegistryError",
    "SLMError",
    "TokenizerError",
]


class SLMError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigError(SLMError):
    """A configuration is missing, malformed, or internally inconsistent."""


class RegistryError(SLMError):
    """A component registry lookup or registration failed."""


class TokenizerError(SLMError):
    """A tokenizer could not be trained, loaded, or applied."""


class DataError(SLMError):
    """A dataset or data-processing step failed."""


class ModelError(SLMError):
    """A model could not be constructed or executed as configured."""


class CheckpointError(SLMError):
    """A checkpoint could not be written, read, or resumed from."""


class EvaluationError(SLMError):
    """A metric or benchmark could not be computed."""
