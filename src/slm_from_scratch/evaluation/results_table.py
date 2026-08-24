"""A results table that exports directly to LaTeX and Markdown.

Every number in the paper traces back to one of these -- generated from a
real evaluation run's JSON output, never typed into the ``.tex`` source by
hand. Regenerating a table from the same results file must reproduce it
byte-for-byte; that's what keeps the paper honest as the codebase evolves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ResultsTable"]


@dataclass
class ResultsTable:
    r"""A rows-of-named-columns results table.

    Attributes:
        caption: Table caption (used in the LaTeX ``\\caption{}``).
        label: LaTeX ``\\label{}`` for cross-referencing.
        rows: ``{row_name: {column_name: value}}``, insertion-ordered.
    """

    caption: str
    label: str
    rows: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_row(self, name: str, values: dict[str, float]) -> None:
        """Add or overwrite one row.

        Args:
            name: Row label (e.g. an ablation rung's id, or a model name).
            values: ``{column_name: value}`` for this row.
        """
        self.rows[name] = dict(values)

    @property
    def columns(self) -> list[str]:
        """Column names, in first-seen order across all rows."""
        seen: dict[str, None] = {}
        for values in self.rows.values():
            for key in values:
                seen[key] = None
        return list(seen)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {"caption": self.caption, "label": self.label, "rows": self.rows}

    def save_json(self, path: str | Path) -> None:
        """Write this table's data as JSON (the source of truth the other formats derive from)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> ResultsTable:
        """Load a table previously written by :meth:`save_json`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        table = cls(caption=data["caption"], label=data["label"])
        table.rows = {name: dict(values) for name, values in data["rows"].items()}
        return table

    def to_markdown(self) -> str:
        """Render as a GitHub-flavored Markdown table."""
        columns = self.columns
        header = "| " + " | ".join(["", *columns]) + " |"
        separator = "|" + "---|" * (len(columns) + 1)
        lines = [header, separator]
        for row_name, values in self.rows.items():
            cells = [_format_value(values.get(col)) for col in columns]
            lines.append("| " + " | ".join([row_name, *cells]) + " |")
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Render as a standalone LaTeX ``table`` environment with a ``tabular`` body."""
        columns = self.columns
        col_spec = "l" + "r" * len(columns)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(["", *(_latex_escape(c) for c in columns)]) + r" \\",
            r"\midrule",
        ]
        for row_name, values in self.rows.items():
            cells = [_format_value(values.get(col)) for col in columns]
            lines.append(" & ".join([_latex_escape(row_name), *cells]) + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{_latex_escape(self.caption)}}}",
            rf"\label{{{self.label}}}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def save_latex(self, path: str | Path) -> None:
        """Write :meth:`to_latex`'s output to ``path`` (typically under ``paper/tables/``)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_latex() + "\n", encoding="utf-8")


def _format_value(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def _latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text
