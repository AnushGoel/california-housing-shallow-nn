"""Landing page: the headline result with error bars, how the model was chosen, and where it misses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import charts
from ..evaluation import interval, paired_bootstrap
from ..theme import DOLLARS, GREY, POPPY, SIGN
from ..uncertainty import conformal_quantile
from .common import fixture_banner, hero, show, signs, usd, usd_html


@st.cache_data(show_spinner=False)
def _mae_interval(y, p, n_boot=1000):
    return interval(paired_bootstrap(y, {"m": p}, n_boot=n_boot, seed=0)["m"]["MAE"], 0.95)


def render(ctx) -> None:
    b = ctx.bundle
    fin, meta, P = b.final, b.meta, b.predictions
    mae_d, rmse_d = fin["test_mae"] * DOLLARS, fin["test_rmse"] * DOLLARS
    hero(f"A {fin['units']}-unit {fin['activation']} network prices California block groups to within about {usd_html(mae_d)}",
         f"Mean absolute error on {b.results['data']['split_sizes']['test']:,} held-out block groups from the 1990 census. "
         f"Every number in this app is computed from the trained artifacts in this repository (fingerprint {b.fingerprint}).")
    fixture_banner(b)
    lo, hi = _mae_interval(P["y_true"].to_numpy(), P["y_pred_final"].to_numpy())
    cards = [("Typical miss (test MAE)", usd_html(mae_d), f"95% interval {usd_html(lo * DOLLARS)} to {usd_html(hi * DOLLARS)}"),
             ("Root-mean-square miss", usd_html(rmse_d), "weights large misses more heavily"),
             ("Variance explained", f"{fin['test_r2']:.2f}", "R² on the test set"),
             ("Weights from epoch", f"{fin['best_epoch']} of {meta['epochs']}", "picked on validation loss")]
    net = b.final_model
    if net.has_intervals:
        q = conformal_quantile(net.cal_residuals, 0.9)
        covered = float(np.mean(np.abs(P["y_true"] - P["y_pred_final"]) <= q))
        cards.insert(1, ("90% prediction interval", f"±{usd_html(q * DOLLARS)}", f"covered {covered:.1%} of test block groups"))
    signs(cards)

    hp = b.results["hpo"]
    trail = [("Step 4: hidden-layer size", f"{b.results['widths']['selected']} units", "lowest mean validation loss over seeds"),
             ("Step 5: activation", b.results["activations"]["selected"], "same rule, ties go to ReLU"),
             ("Step 5b: tuning", f"trial {hp['champion_trial']}", f"validation MSE {hp.get('champion_best_val', float('nan')):.4f}"),
             ("Step 6: test set", usd_html(mae_d), "evaluated once, after every choice")]
    st.markdown("#### How the final model was chosen")
    st.markdown('<div class="trail">' + "".join(f'<div class="stop"><span>{a}</span><b>{v}</b><span>{d}</span></div>'
                                                for a, v, d in trail) + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.35, 1])
    tiles = c1.toggle("Street-map background", value=True, key="ov_tiles")
    data = P.assign(residual=P["y_pred_final"] - P["y_true"])
    show(charts.residual_map(data, "residual", tiles=tiles, title="Where the final model misses (red = over-predicted)",
                             hover={"y_true": ":.2f", "y_pred_final": ":.2f"}), c1)
    comp = pd.DataFrame(b.results["comparison_test"]).rename(columns={"index": "model"}).sort_values("MAE")
    fig = go.Figure(go.Bar(x=(comp["MAE"] * DOLLARS).tolist(), y=comp["model"].tolist(), orientation="h",
                           marker_color=[SIGN if "final model" in m else (POPPY if "extension" in m else GREY) for m in comp["model"]],
                           text=[usd(v * DOLLARS) for v in comp["MAE"]], textposition="outside"))
    fig.update_layout(title="Test MAE in dollars, with reference models", height=400, xaxis_title="mean absolute error ($)",
                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=40, t=56, b=10))
    show(fig, c2)
    c2.markdown('<p class="muted">Reference models were scored on the test set only after every modelling decision had '
                'been made. The Testing page adds error bars and paired comparisons to these numbers.</p>',
                unsafe_allow_html=True)
