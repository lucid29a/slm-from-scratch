"""Tests for ResultsTable."""

from __future__ import annotations

from pathlib import Path

from slm_from_scratch.evaluation.results_table import ResultsTable


def test_columns_reflect_first_seen_order_across_rows() -> None:
    table = ResultsTable(caption="c", label="tab:x")
    table.add_row("a", {"loss": 1.0, "acc": 0.5})
    table.add_row("b", {"ppl": 2.0, "loss": 0.9})
    assert table.columns == ["loss", "acc", "ppl"]


def test_add_row_overwrites_existing_row() -> None:
    table = ResultsTable(caption="c", label="tab:x")
    table.add_row("a", {"loss": 1.0})
    table.add_row("a", {"loss": 0.5})
    assert table.rows["a"] == {"loss": 0.5}


def test_json_round_trip(tmp_path: Path) -> None:
    table = ResultsTable(caption="My Caption", label="tab:my")
    table.add_row("s0", {"loss": 1.416, "ppl": 4.12})
    path = tmp_path / "results.json"
    table.save_json(path)

    loaded = ResultsTable.load_json(path)
    assert loaded.caption == "My Caption"
    assert loaded.label == "tab:my"
    assert loaded.rows == table.rows


def test_to_markdown_contains_row_and_column_labels() -> None:
    table = ResultsTable(caption="c", label="tab:x")
    table.add_row("s0_vanilla", {"loss": 1.416})
    md = table.to_markdown()
    assert "s0_vanilla" in md
    assert "loss" in md
    assert "1.4160" in md


def test_to_markdown_missing_value_renders_as_dash() -> None:
    table = ResultsTable(caption="c", label="tab:x")
    table.add_row("a", {"loss": 1.0})
    table.add_row("b", {"acc": 0.5})
    md = table.to_markdown()
    assert "--" in md


def test_to_latex_contains_environment_and_label() -> None:
    table = ResultsTable(caption="My Caption", label="tab:my")
    table.add_row("s0", {"loss": 1.416})
    latex = table.to_latex()
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex
    assert r"\label{tab:my}" in latex
    assert "My Caption" in latex


def test_to_latex_escapes_special_characters() -> None:
    table = ResultsTable(caption="c", label="tab:x")
    table.add_row("model_a", {"loss": 1.0})
    latex = table.to_latex()
    assert r"model\_a" in latex


def test_save_latex_writes_file(tmp_path: Path) -> None:
    table = ResultsTable(caption="c", label="tab:x")
    table.add_row("s0", {"loss": 1.0})
    path = tmp_path / "tables" / "ablation.tex"
    table.save_latex(path)
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == table.to_latex() + "\n"


def test_regenerating_latex_from_saved_json_is_byte_identical(tmp_path: Path) -> None:
    table = ResultsTable(caption="Ablation results", label="tab:ablation")
    table.add_row("s0_vanilla", {"loss": 1.416, "ppl": 4.12})
    table.add_row("s6_full", {"loss": 0.633, "ppl": 1.88})

    json_path = tmp_path / "results.json"
    table.save_json(json_path)
    first_latex = table.to_latex()

    reloaded = ResultsTable.load_json(json_path)
    second_latex = reloaded.to_latex()

    assert first_latex == second_latex
