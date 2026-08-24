"""A typed component registry.

The registry is the hinge the whole project turns on. Every architectural choice --
which attention, which normalisation, which positional encoding -- is a class
registered under a short string key. A configuration file names the key; the registry
resolves it to a class and builds it.

That indirection is what makes the ablation study in the paper cheap: swapping
``normalization: layernorm`` for ``normalization: rmsnorm`` in a YAML file is the
entire diff between two experiments.

Example:
    >>> class Shape:
    ...     '''Base class for shapes.'''
    >>> SHAPES: Registry[Shape] = Registry("shape", Shape)
    >>> @SHAPES.register("square")
    ... class Square(Shape):
    ...     def __init__(self, side: float) -> None:
    ...         self.side = side
    >>> SHAPES.build("square", side=2.0).side
    2.0
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

from slm_from_scratch.core.exceptions import RegistryError

__all__ = ["Registry"]

T = TypeVar("T")
C = TypeVar("C")


class Registry(Generic[T]):
    """A name-to-class registry constrained to subclasses of a common base.

    Args:
        name: Human-readable name, used in error messages (e.g. ``"attention"``).
        base_class: Every registered class must be a subclass of this.

    Attributes:
        name: The registry's name.
        base_class: The enforced common base class.
    """

    def __init__(self, name: str, base_class: type[T]) -> None:
        self._name = name
        self._base_class = base_class
        self._entries: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(
        self,
        key: str,
        *aliases: str,
    ) -> Callable[[type[C]], type[C]]:
        """Return a decorator registering a class under ``key``.

        Args:
            key: The canonical lookup key.
            *aliases: Additional keys resolving to the same class.

        Returns:
            A class decorator that returns the class unchanged.

        Raises:
            RegistryError: If the key is already taken or the class does not derive
                from this registry's base class.
        """

        def decorator(cls: type[C]) -> type[C]:
            self.register_class(key, cls, *aliases)  # type: ignore[arg-type]
            return cls

        return decorator

    def register_class(self, key: str, cls: type[T], *aliases: str) -> None:
        """Register ``cls`` under ``key`` imperatively.

        Args:
            key: The canonical lookup key.
            cls: The class to register.
            *aliases: Additional keys resolving to the same class.

        Raises:
            RegistryError: If the key is taken or ``cls`` has the wrong base class.
        """
        if not isinstance(cls, type) or not issubclass(cls, self._base_class):
            raise RegistryError(
                f"cannot register {cls!r} in registry {self._name!r}: "
                f"it is not a subclass of {self._base_class.__name__}"
            )
        for name in (key, *aliases):
            if name in self._entries or name in self._aliases:
                raise RegistryError(
                    f"key {name!r} is already registered in registry {self._name!r} "
                    f"to {self.get(name).__name__}"
                )
        self._entries[key] = cls
        for alias in aliases:
            self._aliases[alias] = key

    # ------------------------------------------------------------------ #
    # Lookup and construction
    # ------------------------------------------------------------------ #
    def get(self, key: str) -> type[T]:
        """Resolve ``key`` to a registered class.

        Args:
            key: A canonical key or alias.

        Returns:
            The registered class.

        Raises:
            RegistryError: If the key is unknown. The message suggests near matches,
                because a typo in a config file should not cost you ten minutes.
        """
        resolved = self._aliases.get(key, key)
        try:
            return self._entries[resolved]
        except KeyError:
            raise RegistryError(
                f"unknown {self._name} {key!r}. {self._suggest(key)}"
            ) from None

    def build(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Resolve ``key`` and instantiate it with the given arguments.

        Args:
            key: A canonical key or alias.
            *args: Positional arguments forwarded to the constructor.
            **kwargs: Keyword arguments forwarded to the constructor.

        Returns:
            The constructed instance.

        Raises:
            RegistryError: If the key is unknown or construction fails.
        """
        cls = self.get(key)
        try:
            return cls(*args, **kwargs)
        except TypeError as exc:
            raise RegistryError(
                f"failed to build {self._name} {key!r} ({cls.__name__}): {exc}"
            ) from exc

    def build_from_spec(self, spec: Mapping[str, Any], *args: Any, **kwargs: Any) -> T:
        """Build from a mapping that carries its own type key.

        Args:
            spec: A mapping with a ``"type"`` entry naming the component, plus any
                additional keyword arguments for the constructor.
            *args: Extra positional arguments forwarded to the constructor.
            **kwargs: Extra keyword arguments, overridden by those in ``spec``.

        Returns:
            The constructed instance.

        Raises:
            RegistryError: If ``spec`` has no ``"type"`` entry.
        """
        params = dict(spec)
        key = params.pop("type", None)
        if not isinstance(key, str):
            raise RegistryError(
                f"registry {self._name!r} needs a string 'type' entry in the spec, got {spec!r}"
            )
        return self.build(key, *args, **{**kwargs, **params})

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """The registry's human-readable name."""
        return self._name

    @property
    def base_class(self) -> type[T]:
        """The base class every entry must derive from."""
        return self._base_class

    def keys(self) -> tuple[str, ...]:
        """Return the canonical keys, sorted."""
        return tuple(sorted(self._entries))

    def items(self) -> tuple[tuple[str, type[T]], ...]:
        """Return ``(key, class)`` pairs, sorted by key."""
        return tuple((k, self._entries[k]) for k in self.keys())

    def _suggest(self, key: str) -> str:
        """Build a 'did you mean' fragment for an unknown key."""
        known = [*self._entries, *self._aliases]
        close = difflib.get_close_matches(key, known, n=3)
        if close:
            return f"Did you mean {' or '.join(repr(c) for c in close)}?"
        return f"Available: {', '.join(repr(k) for k in self.keys()) or '(none registered)'}"

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self._entries or key in self._aliases)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={list(self.keys())})"
