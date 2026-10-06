"""REST API contract tests (run in CI, where FastAPI and httpx are installed)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(artifacts_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    from api import main
    main.service.cache_clear()
    return TestClient(main.app)


def test_health_reports_the_verified_fingerprint(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and len(body["fingerprint"]) == 12 and body["checksums_verified"]


def test_predict_returns_ordered_intervals(client):
    r = client.post("/predict", json={"block_groups": [{"MedInc": 8.3, "Latitude": 37.8, "Longitude": -122.3}, {}], "level": 0.9})
    assert r.status_code == 200
    for p in r.json()["predictions"]:
        assert p["lower"] <= p["prediction"] <= p["upper"]


def test_implausible_input_is_rejected(client):
    assert client.post("/predict", json={"block_groups": [{"MedInc": -1}]}).status_code == 422
    assert client.post("/predict", json={"block_groups": []}).status_code == 422


def test_explanation_adds_up(client):
    body = client.post("/explain", json={"MedInc": 9.0}).json()
    assert abs(body["baseline"] + sum(c["contribution"] for c in body["contributions"]) - body["prediction"]) < 1e-6
