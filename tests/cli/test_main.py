"""Tests for the `slm` argparse root parser and dispatch."""

from __future__ import annotations

import pytest

from slm_from_scratch.cli.main import build_parser, main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert "slm-from-scratch" in capsys.readouterr().out


def test_no_command_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_unknown_command_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main(["not-a-real-command"])


def test_every_registered_command_gets_a_subparser() -> None:
    parser = build_parser()
    subparsers_action = next(a for a in parser._actions if a.dest == "command")
    choices = subparsers_action.choices
    assert choices is not None
    assert "train" in choices
    assert "generate" in choices


def test_help_does_not_raise() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
