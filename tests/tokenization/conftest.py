"""Shared fixtures for tokenization tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def toy_corpus() -> list[str]:
    return [
        "the quick brown fox jumps over the lazy dog",
        "the dog barks at the fox in the night",
        "quick foxes are quicker than slow dogs",
        "a lazy dog sleeps while the quick fox runs",
    ]
