"""Regenerate docs/RESULTS.md, docs/figures/ and the README results block from the artifacts.

    python scripts/build_report.py            # same as: python -m housing_app report
    python scripts/build_report.py --check    # exit 1 if the report does not match the artifacts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from housing_app.cli import main  # noqa: E402

raise SystemExit(main(["report", *sys.argv[1:]]))
