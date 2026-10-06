#!/usr/bin/env python3
"""Build the research notebook from its plain-text sources ("notebook as code").

The files in notebooks/source/ are diff-friendly text: a line `# %% [markdown]` starts a Markdown cell and `# %%`
starts a code cell. Reviews happen on the text; outputs never live in git (run the built notebook to create them).

    python scripts/build_notebook.py                         # -> notebooks/california_housing_shallow_nn.ipynb
    python scripts/build_notebook.py --edition my.private.json --out other.ipynb

An edition file rewords the notebook without forking it: {"title": "...", "replacements": [[from, to, count], ...]}.
Keep personal editions out of git (the .gitignore excludes *.private.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebooks" / "source"
DEFAULT_OUT = ROOT / "notebooks" / "california_housing_shallow_nn.ipynb"
DEFAULT_TITLE = "California housing: a shallow neural network, studied carefully"


def read_sources(source_dir: Path = SOURCE_DIR) -> str:
    parts = sorted(source_dir.glob("part*.txt"))
    if not parts:
        raise SystemExit(f"no notebook sources found in {source_dir}")
    return "\n".join(p.read_text(encoding="utf-8") for p in parts)


def apply_edition(text: str, edition: dict) -> str:
    for old, new, count in edition.get("replacements", []):
        found = text.count(old)
        if found != count:
            raise SystemExit(f"edition does not match the sources: {old[:70]!r} occurs {found}x, expected {count}x")
        text = text.replace(old, new)
    return text


def parse_cells(text: str) -> list:
    cells, kind, buf = [], None, []
    for line in text.splitlines() + ["# %%"]:
        mark = line.strip()
        if mark in ("# %% [markdown]", "# %%"):
            if kind is not None and "\n".join(buf).strip():
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("markdown" if mark.endswith("[markdown]") else "code"), []
        else:
            buf.append(line)
    return cells


def to_notebook(cells: list, title: str) -> dict:
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        if kind == "code":
            compile(src, f"<cell {i}>", "exec")              # fail fast on syntax errors
        lines = src.split("\n")
        cell = {"cell_type": kind, "id": f"c{i:03d}", "metadata": {},
                "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"title": title, "authors": [{"name": "Anush Goel"}],
                         "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "pygments_lexer": "ipython3"}}}


def build(out: Path = DEFAULT_OUT, edition_path=None) -> dict:
    text, title = read_sources(), DEFAULT_TITLE
    if edition_path:
        edition = json.loads(Path(edition_path).read_text(encoding="utf-8"))
        text, title = apply_edition(text, edition), edition.get("title", title)
    nb = to_notebook(parse_cells(text), title)
    Path(out).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--edition", type=Path, help="optional JSON file that rewords the notebook")
    args = parser.parse_args()
    nb = build(args.out, args.edition)
    print(f"built {args.out}: {len(nb['cells'])} cells ({sum(c['cell_type'] == 'code' for c in nb['cells'])} code)")


if __name__ == "__main__":
    main()
