"""The `slm` command-line tool: one Command subclass per subcommand, in a registry."""

from __future__ import annotations

from slm_from_scratch.cli.base import COMMANDS, Command

__all__ = ["COMMANDS", "Command"]
