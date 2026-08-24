"""The command contract every `slm` subcommand implements.

Adding a new subcommand means writing a new :class:`Command` subclass and
registering it -- :mod:`slm_from_scratch.cli.main` never special-cases which
commands exist; it just asks the registry.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar

from slm_from_scratch.core.registry import Registry

__all__ = ["COMMANDS", "Command"]


class Command(ABC):
    """Abstract base for one ``slm <name> ...`` subcommand.

    Attributes:
        name: The subcommand name, e.g. ``"train"`` for ``slm train``.
        help: One-line help text shown in ``slm --help``.
    """

    name: ClassVar[str]
    help: ClassVar[str] = ""

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register this command's CLI arguments on ``parser``."""
        raise NotImplementedError

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """Execute the command.

        Args:
            args: Parsed arguments, as populated by :meth:`add_arguments`.

        Returns:
            A process exit code (``0`` for success).
        """
        raise NotImplementedError


COMMANDS: Registry[Command] = Registry("command", Command)  # type: ignore[type-abstract]
