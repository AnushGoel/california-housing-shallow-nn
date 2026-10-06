import numpy as np
import pytest

from housing_app.uncertainty import conformal_quantile, coverage, coverage_table


def test_quantile_uses_the_finite_sample_rank():
    scores = np.arange(1.0, 11.0)                        # n = 10
    assert conformal_quantile(scores, 0.9) == 10.0       # ceil(11 * 0.9) = 10th smallest
    assert conformal_quantile(scores, 0.5) == 6.0        # ceil(11 * 0.5) = 6th smallest
    assert conformal_quantile(scores[:3], 0.9) == float("inf")      # too few scores for 90%


def test_coverage_is_at_least_the_promise_on_exchangeable_data():
    rng = np.random.default_rng(0)
    achieved = []
    for _ in range(200):
        noise = rng.standard_t(df=3, size=400)          # heavy-tailed: no Gaussian assumption needed
        cal, test = np.abs(noise[:200]), noise[200:]
        achieved.append(coverage(test, np.zeros(200), cal, 0.9))
    assert np.mean(achieved) >= 0.89


def test_coverage_table_reports_groups():
    y, p = np.array([0.0, 1.0, 2.0, 3.0]), np.zeros(4)
    table = coverage_table(y, p, scores=np.array([0.5, 1.5, 2.5]), levels=(0.5,), groups=np.array(list("aabb")))
    assert {"promised", "half_width", "achieved", "a", "b"} <= set(table.columns)


def test_invalid_level_is_rejected():
    with pytest.raises(ValueError):
        conformal_quantile([1.0], 1.0)


def test_exported_models_carry_calibration(bundle):
    net = bundle.final_model
    lo, hi = net.interval(bundle.predictions[bundle.features].to_numpy()[:5], 0.9)
    assert net.has_intervals and np.all(lo < hi)
