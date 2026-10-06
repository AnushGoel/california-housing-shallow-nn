"""Copy the notebook's artifacts (and their manifest) to the configured S3-compatible bucket.

    python scripts/publish_artifacts.py       # same as: python -m housing_app publish
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from housing_app.cli import main  # noqa: E402

raise SystemExit(main(["publish", *sys.argv[1:]]))
