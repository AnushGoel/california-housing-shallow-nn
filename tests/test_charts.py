"""Chart builders return valid Plotly figures for the fixture data (skipped where Plotly is not installed)."""
import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects")
from housing_app import charts  # noqa: E402


def relu_runs(bundle):
    return [(n, r) for n, r in bundle.histories.items() if r["cfg"].get("lr") is None and r["cfg"]["activation"] == "relu"]


def test_learning_curves_and_animation(bundle):
    runs = relu_runs(bundle)
    fig, gap = charts.learning_curves(runs, "hidden_units", {}, " units", bundle.meta["seed"])
    assert isinstance(fig, go.Figure) and fig.data and gap.data
    anim = charts.animated_learning(runs, "hidden_units", {}, " units", bundle.meta["seed"])
    assert anim is not None and len(anim.frames) > 1


def test_tuning_and_testing_charts(bundle):
    trials = pd.DataFrame(bundle.results["hpo"]["trials"])
    assert len(charts.parcoords(trials).data[0].dimensions) == 7
    assert isinstance(charts.hp_scatter(trials, "lr", 0.3), go.Figure)
    cv = pd.DataFrame(bundle.results["evaluation"]["cv"])
    assert isinstance(charts.cv_dots(cv, bundle.final["test_mae"]), go.Figure)
    df = pd.DataFrame({"model": ["a", "b"], "est": [1.0, 2.0], "lo": [0.5, 1.5], "hi": [1.5, 2.5]})
    assert len(charts.forest(df, "model", "est", "lo", "hi", title="t", xaxis_title="x").data) == 4


def test_maps_and_diagnostics(bundle):
    P = bundle.predictions.assign(residual=bundle.predictions["y_pred_final"] - bundle.predictions["y_true"])
    for tiles in (True, False):
        assert isinstance(charts.residual_map(P, "residual", tiles=tiles), go.Figure)
        assert isinstance(charts.pin_map(37.0, -120.0, context=P, tiles=tiles), go.Figure)
    assert isinstance(charts.fold_map(bundle.dataset), go.Figure)
    assert isinstance(charts.parity(P["y_true"], P["y_pred_final"]), go.Figure)
    assert isinstance(charts.decile_bars(P["y_true"], P["y_pred_final"]), go.Figure)
