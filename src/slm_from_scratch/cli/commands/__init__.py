"""Concrete `slm` subcommands. Importing this package registers all of them."""

from __future__ import annotations

from slm_from_scratch.cli.commands.ablate import AblateCommand
from slm_from_scratch.cli.commands.generate import GenerateCommand
from slm_from_scratch.cli.commands.gui import GuiCommand
from slm_from_scratch.cli.commands.pack import PackCommand
from slm_from_scratch.cli.commands.prepare_data import PrepareDataCommand
from slm_from_scratch.cli.commands.train import TrainCommand
from slm_from_scratch.cli.commands.train_tokenizer import TrainTokenizerCommand

__all__ = [
    "AblateCommand",
    "GenerateCommand",
    "GuiCommand",
    "PackCommand",
    "PrepareDataCommand",
    "TrainCommand",
    "TrainTokenizerCommand",
]
