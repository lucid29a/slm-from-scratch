"""Small helpers shared across CLI commands."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from slm_from_scratch.core.exceptions import ConfigError

__all__ = ["load_yaml_mapping", "read_corpus"]


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require it to be a mapping at the top level.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.

    Raises:
        ConfigError: If the file is missing, malformed, or not a mapping.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")
    try:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{file_path} must contain a mapping at the top level")
    return data


def read_corpus(path: str | Path) -> Iterator[str]:
    """Yield documents from a JSONL file (``{"text": ...}`` per line) or plain text lines.

    Args:
        path: A file produced by ``slm prepare-data`` (JSONL), or any plain-text
            file where each non-blank line is one document.

    Yields:
        Document strings.
    """
    with Path(path).open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)["text"] if line.startswith("{") else line
