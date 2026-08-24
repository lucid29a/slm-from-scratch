"""Unit tests for slm_from_scratch.core.component.Component."""

from __future__ import annotations

from dataclasses import dataclass

from slm_from_scratch.core.component import Component
from slm_from_scratch.core.config import BaseConfig


@dataclass(frozen=True, kw_only=True)
class GreeterConfig(BaseConfig):
    greeting: str = "hello"


class Greeter(Component[GreeterConfig]):
    def greet(self, name: str) -> str:
        return f"{self.config.greeting}, {name}!"


def test_component_exposes_its_config() -> None:
    cfg = GreeterConfig(greeting="hi")
    greeter = Greeter(cfg)
    assert greeter.config is cfg
    assert greeter.greet("world") == "hi, world!"


def test_component_repr_includes_config() -> None:
    greeter = Greeter(GreeterConfig())
    assert "Greeter" in repr(greeter)
    assert "hello" in repr(greeter)
