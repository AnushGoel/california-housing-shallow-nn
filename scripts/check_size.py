"""Keep the repository light: fail if tracked data folders or single files grow past their budgets.

    python scripts/check_size.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET_MB = {"artifacts": 8.0, "tests/fixtures": 1.0, "docs": 3.0, "notebooks": 2.0, "results": 10.0}
FILE_LIMIT_MB = 3.0
FILE_LIMIT_OVERRIDES_MB = {"results": 8.0}        # the executed notebook carries 27 figures
IGNORED = {"runs", "__pycache__", ".ipynb_checkpoints"}


def folder_files(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and not (IGNORED & set(p.relative_to(ROOT).parts))]


def main() -> int:
    failed = False
    print(f"{'folder':<16}{'size':>10}{'budget':>10}")
    for name, budget in BUDGET_MB.items():
        files = folder_files(ROOT / name) if (ROOT / name).exists() else []
        size = sum(p.stat().st_size for p in files) / 1e6
        over = size > budget
        failed |= over
        print(f"{name:<16}{size:>8.2f}MB{budget:>8.1f}MB{'  OVER BUDGET' if over else ''}")
        limit = FILE_LIMIT_OVERRIDES_MB.get(name, FILE_LIMIT_MB)
        for p in files:
            if p.stat().st_size / 1e6 > limit:
                failed = True
                print(f"  file too large ({p.stat().st_size / 1e6:.1f} MB > {limit} MB): {p.relative_to(ROOT)}")
    print("size budget: FAILED" if failed else "size budget: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
