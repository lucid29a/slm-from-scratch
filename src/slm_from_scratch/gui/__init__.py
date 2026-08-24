"""A local Gradio GUI: chat with a checkpoint, monitor a training run live.

Requires the optional ``gui`` extra (``pip install -e ".[gui]"``).
"""

from __future__ import annotations

__all__ = ["build_app"]


def __getattr__(name: str) -> object:
    # Deferred import: gradio is an optional dependency and importing it
    # shouldn't be a cost paid by every `import slm_from_scratch`.
    if name == "build_app":
        from slm_from_scratch.gui.app import build_app

        return build_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
