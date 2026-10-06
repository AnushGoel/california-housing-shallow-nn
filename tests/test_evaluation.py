import numpy as np

from housing_app.evaluation import compare, interval, mae, nearest_place, paired_bootstrap, r2, rmse, slice_table


def test_metrics_match_their_definitions():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 2.0])
    assert mae(y, p) == 1.0 and rmse(y, p) == np.sqrt(5 / 3)
    assert abs(r2(y, p) - (1 - 5 / np.sum((y - y.mean()) ** 2))) < 1e-12


def test_bootstrap_interval_contains_the_estimate_and_shrinks_with_more_rows():
    rng = np.random.default_rng(1)
    widths = []
    for n in (200, 3200):
        y = rng.normal(size=n)
        p = y + rng.normal(scale=0.5, size=n)
        lo, hi = interval(paired_bootstrap(y, {"m": p}, n_boot=600, seed=2)["m"]["MAE"])
        assert lo <= mae(y, p) <= hi
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 2


def test_paired_comparison_detects_a_clearly_better_model():
    rng = np.random.default_rng(3)
    y = rng.normal(size=800)
    draws = paired_bootstrap(y, {"good": y + rng.normal(scale=0.1, size=800),
                                 "bad": y + rng.normal(scale=1.0, size=800)}, n_boot=500)
    result = compare(draws["good"], draws["bad"], "MAE")
    assert result["verdict"] == "first better" and result["high"] < 0 and result["p_two_sided"] < 0.01
    assert compare(draws["good"], draws["bad"], "R2")["verdict"] == "first better"


def test_identical_models_show_no_difference():
    draws = paired_bootstrap(np.arange(50.0), {"m": np.arange(50.0) + 1}, n_boot=200)["m"]
    result = compare(draws, draws)
    assert result["difference"] == 0 and result["verdict"] == "no clear difference" and result["p_two_sided"] == 1


def test_slice_table_counts_rows_and_puts_the_worst_slice_first():
    y = np.zeros(6)
    p = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    table = slice_table(y, p, np.array(list("aaabbb")), n_boot=100)
    assert table["slice"].tolist() == ["a", "b"] and table["n"].tolist() == [3, 3]
    assert table.loc[0, "MAE"] == 1.0


def test_nearest_place():
    places = {"north": (40.0, -120.0), "south": (33.0, -117.0)}
    assert nearest_place([39.5, 33.2], [-121.0, -117.5], places).tolist() == ["north", "south"]
