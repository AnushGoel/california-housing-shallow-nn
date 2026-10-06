"""End-to-end: render every page of the real Streamlit app against the synthetic fixtures (runs in CI)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_every_page_renders_without_errors(artifacts_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("HOUSING_USER_DIR", str(tmp_path / "user_data"))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    options = at.radio(key="nav").options
    assert len(options) == 10
    for page in options:
        at.radio(key="nav").set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"


def test_missing_artifacts_show_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "nothing_here"))
    monkeypatch.setenv("HOUSING_USER_DIR", str(tmp_path / "user_data"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("could not be loaded" in element.value for element in at.error)
