"""Tests for the Command registry and dispatch."""

from __future__ import annotations

from slm_from_scratch.cli.base import COMMANDS


def test_all_expected_commands_are_registered() -> None:
    expected = {"prepare-data", "train-tokenizer", "pack", "train", "generate", "ablate"}
    assert expected <= set(COMMANDS.keys())


def test_registered_commands_expose_name_and_help() -> None:
    for key, command_cls in COMMANDS.items():
        command = command_cls()
        assert command.name == key
        assert isinstance(command.help, str)
