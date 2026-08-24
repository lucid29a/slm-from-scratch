"""Unit tests for slm_from_scratch.core.registry.Registry."""

from __future__ import annotations

import pytest

from slm_from_scratch.core.exceptions import RegistryError
from slm_from_scratch.core.registry import Registry


class Animal:
    """Common base for the fixtures below."""


class Dog(Animal):
    def __init__(self, name: str = "Rex") -> None:
        self.name = name


class Cat(Animal):
    pass


class NotAnAnimal:
    pass


@pytest.fixture
def registry() -> Registry[Animal]:
    return Registry("animal", Animal)


def test_register_and_build(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    animal = registry.build("dog", name="Fido")
    assert isinstance(animal, Dog)
    assert animal.name == "Fido"


def test_register_via_decorator() -> None:
    reg: Registry[Animal] = Registry("animal", Animal)

    @reg.register("cat")
    class RegisteredCat(Cat):
        pass

    assert reg.build("cat").__class__.__name__ == "RegisteredCat"


def test_register_wrong_base_class_raises(registry: Registry[Animal]) -> None:
    with pytest.raises(RegistryError, match="not a subclass"):
        registry.register_class("nope", NotAnAnimal)  # type: ignore[arg-type]


def test_duplicate_key_raises(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register_class("dog", Dog)


def test_unknown_key_suggests_close_match(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    with pytest.raises(RegistryError, match="Did you mean 'dog'"):
        registry.build("dgo")


def test_unknown_key_without_close_match_lists_available(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    with pytest.raises(RegistryError, match="Available: 'dog'"):
        registry.build("zzz_completely_unrelated")


def test_aliases_resolve_to_canonical(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog, "puppy", "hound")
    assert isinstance(registry.build("puppy"), Dog)
    assert isinstance(registry.build("hound"), Dog)


def test_build_from_spec(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    animal = registry.build_from_spec({"type": "dog", "name": "Buddy"})
    assert isinstance(animal, Dog)
    assert animal.name == "Buddy"


def test_build_from_spec_requires_type_key(registry: Registry[Animal]) -> None:
    with pytest.raises(RegistryError, match="needs a string 'type'"):
        registry.build_from_spec({"name": "Buddy"})


def test_build_construction_error_is_wrapped(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    with pytest.raises(RegistryError, match="failed to build"):
        registry.build("dog", unexpected_kwarg=123)


def test_contains_len_iter_keys(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog, "puppy")
    registry.register_class("cat", Cat)
    assert "dog" in registry
    assert "puppy" in registry
    assert "bird" not in registry
    assert len(registry) == 2
    assert list(registry) == ["cat", "dog"]
    assert registry.keys() == ("cat", "dog")


def test_items_returns_sorted_pairs(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    registry.register_class("cat", Cat)
    assert registry.items() == (("cat", Cat), ("dog", Dog))


def test_repr_lists_keys(registry: Registry[Animal]) -> None:
    registry.register_class("dog", Dog)
    assert "dog" in repr(registry)
