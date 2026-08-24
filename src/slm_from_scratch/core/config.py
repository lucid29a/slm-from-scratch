"""Configuration objects: frozen dataclasses that load from YAML and validate themselves.

Every subsystem in this project is driven by a config, never by hand-written keyword
arguments scattered through call sites. Configs are:

* **Frozen** -- a config that was used to build a model cannot silently drift while
  training runs, which matters for reproducibility.
* **Self-validating** -- ``__post_init__`` catches nonsensical values (negative
  dimensions, a head count that does not divide the model dimension) at construction
  time, not three hundred training steps later.
* **Composable** -- a YAML file may ``!extends`` another, so ``modern_150m.yaml`` can
  say "everything in modern_50m.yaml, but ``n_layer: 18``" instead of repeating
  every field.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import yaml

from slm_from_scratch.core.exceptions import ConfigError

__all__ = ["BaseConfig", "load_yaml_with_extends"]

_EXTENDS_KEY = "extends"

ConfigT = TypeVar("ConfigT", bound="BaseConfig")


def load_yaml_with_extends(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving a top-level ``extends: other.yaml`` chain.

    ``extends`` is resolved relative to the file that declares it. Keys in the child
    override keys in the parent; nested mappings are merged recursively so a child
    can override a single field of a nested block without repeating the rest.

    Args:
        path: Path to the YAML file to load.

    Returns:
        The fully merged mapping, with the ``extends`` key removed.

    Raises:
        ConfigError: If the file is missing, is not a mapping at the top level, or
            an ``extends`` chain is circular.
    """
    return _load_with_extends(Path(path), seen=set())


def _load_with_extends(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ConfigError(f"circular 'extends' chain: {chain}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")

    parent_name = raw.pop(_EXTENDS_KEY, None)
    if parent_name is None:
        return raw

    parent_path = (path.parent / parent_name).resolve()
    parent = _load_with_extends(parent_path, seen={*seen, resolved})
    return _deep_merge(parent, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, kw_only=True)
class BaseConfig:
    """Base class for every configuration dataclass in the project.

    Subclasses are plain ``@dataclass(frozen=True, kw_only=True)`` bodies; this base
    supplies YAML loading, dict round-tripping, and a validation hook.
    """

    #: Set by subclasses that want stricter loading; unknown YAML keys otherwise
    #: raise, since a silently-ignored typo in a config is a training run wasted.
    _strict: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Run subclass validation immediately after construction.

        Raises:
            ConfigError: If :meth:`validate` finds a problem.
        """
        self.validate()

    def validate(self) -> None:
        """Check invariants beyond what the type system expresses.

        The base implementation does nothing; override in subclasses and call
        ``super().validate()`` first if you want the (currently empty) base checks.

        Raises:
            ConfigError: If an invariant is violated.
        """

    @classmethod
    def from_dict(cls: type[ConfigT], data: dict[str, Any]) -> ConfigT:
        """Construct from a plain mapping, e.g. one parsed from YAML or JSON.

        Args:
            data: Field values, keyed by field name.

        Returns:
            A validated instance.

        Raises:
            ConfigError: If ``data`` has unknown keys (when strict) or fails to
                construct the dataclass.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown and cls._strict:
            raise ConfigError(
                f"unknown field(s) for {cls.__name__}: {sorted(unknown)}. "
                f"Known fields: {sorted(known)}"
            )
        payload = {k: v for k, v in data.items() if k in known}
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"failed to build {cls.__name__} from {data!r}: {exc}") from exc

    @classmethod
    def from_yaml(cls: type[ConfigT], path: str | Path) -> ConfigT:
        """Load a config from a YAML file, resolving any ``extends`` chain.

        Args:
            path: Path to a YAML file.

        Returns:
            A validated instance.
        """
        return cls.from_dict(load_yaml_with_extends(path))

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied plain-dict representation."""
        return copy.deepcopy(asdict(self))

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file.

        Args:
            path: Destination path; parent directories are created if needed.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def replace(self: ConfigT, **overrides: Any) -> ConfigT:
        """Return a copy with the given fields overridden (dataclasses.replace, validated).

        Args:
            **overrides: Field values to override.

        Returns:
            A new, validated instance.
        """
        import dataclasses

        return dataclasses.replace(self, **overrides)
