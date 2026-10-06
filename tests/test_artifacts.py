import json

import pandas as pd
import pytest

from housing_app.artifacts import ArtifactError, load_bundle
from housing_app.storage import LocalStorage


def test_fixture_bundle_loads_and_is_flagged_as_a_fixture(bundle):
    assert bundle.is_test_fixture and len(bundle.features) == 8 and bundle.has_evaluation and bundle.verified
    assert {"Final model", "Tuned champion"} <= set(bundle.prediction_columns())


def test_missing_file_is_reported_by_name(artifacts_dir):
    (artifacts_dir / "histories.json.gz").unlink()
    with pytest.raises(ArtifactError, match="histories.json"):
        load_bundle(LocalStorage(artifacts_dir))


def test_results_without_a_required_section_are_rejected(artifacts_dir):
    (artifacts_dir / "manifest.json").unlink()
    path = artifacts_dir / "results.json"
    results = json.loads(path.read_text())
    del results["final"]
    path.write_text(json.dumps(results))
    with pytest.raises(ArtifactError, match="'final'"):
        load_bundle(LocalStorage(artifacts_dir))


def test_predictions_without_the_final_model_column_are_rejected(artifacts_dir):
    (artifacts_dir / "manifest.json").unlink()
    path = artifacts_dir / "test_predictions.csv.gz"
    pd.read_csv(path).drop(columns=["y_pred_final"]).to_csv(path, index=False, compression="gzip")
    with pytest.raises(ArtifactError, match="y_pred_final"):
        load_bundle(LocalStorage(artifacts_dir))


def test_altered_file_fails_the_checksum(artifacts_dir):
    path = artifacts_dir / "results.json"
    path.write_text(path.read_text().replace('"seed"', '"seed" ', 1))
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_bundle(LocalStorage(artifacts_dir))


def test_uncompressed_artifacts_load_too(artifacts_dir):
    import gzip
    (artifacts_dir / "manifest.json").unlink()
    for name in ("histories.json", "dataset.csv", "test_predictions.csv"):
        gz = artifacts_dir / f"{name}.gz"
        (artifacts_dir / name).write_bytes(gzip.decompress(gz.read_bytes()))
        gz.unlink()
    bundle = load_bundle(LocalStorage(artifacts_dir))
    assert not bundle.verified and len(bundle.fingerprint) == 12


def test_corrupt_json_is_reported_not_crashed(artifacts_dir):
    (artifacts_dir / "manifest.json").unlink()
    (artifacts_dir / "results.json").write_text("{not json")
    with pytest.raises(ArtifactError, match="could not read"):
        load_bundle(LocalStorage(artifacts_dir))
