import shutil
from pathlib import Path

import pytest

from housing_app.artifacts import load_bundle
from housing_app.storage import LocalStorage

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_artifacts"


@pytest.fixture
def artifacts_dir(tmp_path):
    """A writable copy of the synthetic artifact bundle (produced by the notebook's own export code)."""
    target = tmp_path / "artifacts"
    shutil.copytree(FIXTURE_DIR, target)
    return target


@pytest.fixture
def bundle(artifacts_dir):
    return load_bundle(LocalStorage(artifacts_dir))
