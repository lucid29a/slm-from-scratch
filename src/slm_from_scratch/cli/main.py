"""The ``slm`` command-line entry point.

Builds an ``argparse`` subcommand for every :class:`~slm_from_scratch.cli.base.Command`
registered in ``COMMANDS`` -- adding a new subcommand never touches this file.
"""

from __future__ import annotations

import argparse
import contextlib
import sys

from slm_from_scratch import __version__
from slm_from_scratch.cli import commands as _commands  # noqa: F401  (registers commands)
from slm_from_scratch.cli.base import COMMANDS
from slm_from_scratch.core.exceptions import SLMError

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the root ``slm`` parser with one subparser per registered command."""
    parser = argparse.ArgumentParser(prog="slm", description="slm-from-scratch command-line tool.")
    parser.add_argument("--version", action="version", version=f"slm-from-scratch {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for key, command_cls in COMMANDS.items():
        command = command_cls()
        sub = subparsers.add_parser(key, help=command.help)
        command.add_arguments(sub)
        sub.set_defaults(_command=command)

    return parser


def _warm_optional_imports() -> None:
    """Import heavy optional dependencies eagerly, before any command runs.

    ``datasets`` pulls in ``pyarrow``, whose native extension does its own
    filesystem scanning on first import. Importing it lazily, deep inside a
    generator, races on Windows with concurrent filesystem activity (this
    project hit ``OSError: [WinError 6714]`` from exactly that race); doing it
    once, up front, before any network I/O starts, avoids the race entirely.
    Not installed (the ``data`` extra) is a normal, silent case.
    """
    with contextlib.suppress(ImportError):
        import datasets  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    _warm_optional_imports()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args._command.run(args))
    except SLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
