import numpy as np
import pandas as pd

from housing_app.monitoring import build_reference, drift_table, ks_statistic, psi, validate_frame

FEATURES = ["a", "b"]


def test_validation_reports_missing_columns_bad_values_and_implausible_rows():
    _, ok, problems = validate_frame(pd.DataFrame({"a": [1.0]}), FEATURES)
    assert "missing required columns: b" in problems[0] and not ok.any()
    df = pd.DataFrame({"MedInc": [3.0, "x", 99.0], "HouseAge": [10, 20, 30]})
    _, ok, problems = validate_frame(df, ["MedInc", "HouseAge"])
    assert ok.tolist() == [True, False, False] and len(problems) == 2


def test_psi_is_zero_for_identical_and_large_for_shifted_distributions():
    e = np.full(10, 0.1)
    assert psi(e, e) == 0.0
    assert psi(e, np.r_[np.full(5, 0.02), np.full(5, 0.18)]) > 0.25


def test_ks_statistic_matches_a_hand_computed_case():
    assert ks_statistic(np.array([1.0, 2.0, 3.0, 4.0]), [3.5, 4.5]) == 0.75


def test_reference_flags_far_away_rows_and_detects_drift():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    ref = build_reference(train, lambda X: np.asarray(X, float))
    same = drift_table(ref, pd.DataFrame({"a": rng.normal(size=800), "b": rng.normal(size=800)}))
    moved = drift_table(ref, pd.DataFrame({"a": rng.normal(1.5, 1, size=800), "b": rng.normal(size=800)}))
    assert set(same["status"]) == {"stable"}
    assert moved.iloc[0]["feature"] == "a" and moved.iloc[0]["status"] == "major shift"
    assert 0.5 < ref.threshold < 6
