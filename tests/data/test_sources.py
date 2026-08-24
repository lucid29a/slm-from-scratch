"""Tests for text sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_from_scratch.core.exceptions import DataError
from slm_from_scratch.data.sources import LocalFileSource, LocalFileSourceConfig


def test_local_file_source_one_doc_per_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("first document", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second document", encoding="utf-8")

    src = LocalFileSource(LocalFileSourceConfig(root=str(tmp_path)))
    docs = sorted(src)
    assert docs == ["first document", "second document"]


def test_local_file_source_one_doc_per_line(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("line one\nline two\n\nline three", encoding="utf-8")

    src = LocalFileSource(LocalFileSourceConfig(root=str(tmp_path), one_doc_per_line=True))
    docs = list(src)
    assert docs == ["line one", "line two", "line three"]


def test_local_file_source_respects_pattern(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("keep me", encoding="utf-8")
    (tmp_path / "skip.md").write_text("skip me", encoding="utf-8")

    src = LocalFileSource(LocalFileSourceConfig(root=str(tmp_path), pattern="*.txt"))
    assert list(src) == ["keep me"]


def test_local_file_source_respects_limit(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"doc_{i}.txt").write_text(f"document {i}", encoding="utf-8")

    src = LocalFileSource(LocalFileSourceConfig(root=str(tmp_path), limit=2))
    assert len(list(src)) == 2


def test_local_file_source_missing_root_raises(tmp_path: Path) -> None:
    src = LocalFileSource(LocalFileSourceConfig(root=str(tmp_path / "nonexistent")))
    with pytest.raises(DataError, match="not a directory"):
        list(src)


def test_local_file_source_skips_blank_files(tmp_path: Path) -> None:
    (tmp_path / "blank.txt").write_text("   \n  ", encoding="utf-8")
    (tmp_path / "real.txt").write_text("real content", encoding="utf-8")

    src = LocalFileSource(LocalFileSourceConfig(root=str(tmp_path)))
    assert list(src) == ["real content"]
