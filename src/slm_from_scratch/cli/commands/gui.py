"""``slm gui``: launch the local Gradio chat + training-dashboard app."""

from __future__ import annotations

import argparse

from slm_from_scratch.cli.base import COMMANDS, Command
from slm_from_scratch.core.exceptions import SLMError

__all__ = ["GuiCommand"]


@COMMANDS.register("gui")
class GuiCommand(Command):
    """Launch the local web GUI (chat with a checkpoint, monitor training runs)."""

    name = "gui"
    help = "Launch the local Gradio GUI (chat + training dashboard)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register host/port/share arguments."""
        parser.add_argument("--host", default="127.0.0.1", help="Interface to bind to.")
        parser.add_argument("--port", type=int, default=7860, help="Port to serve on.")
        parser.add_argument(
            "--share", action="store_true", help="Create a public Gradio share link."
        )

    def run(self, args: argparse.Namespace) -> int:
        """Build and launch the Gradio app."""
        try:
            from slm_from_scratch.gui.app import build_app
        except ImportError as exc:
            raise SLMError(
                "slm gui requires the 'gui' extra; install with `pip install -e '.[gui]'`"
            ) from exc

        app = build_app()
        app.launch(server_name=args.host, server_port=args.port, share=args.share)
        return 0
