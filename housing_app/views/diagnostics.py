"""Diagnostics: parity, error by price decile, a selectable error map, the price ceiling and feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts
from ..evaluation import mae
from ..theme import DOLLARS
from .common import page_title, selected_indices, show, signs, table, usd


def render(ctx) -> None:
    b = ctx.bundle
    P, cols = b.predictions, b.prediction_columns()
    page_title("Diagnostics", "Where and how the models err on the test set. Draw a box or lasso on the map to inspect any area.")
    which = st.radio("Model", list(cols), horizontal=True, key="diag_model")
    true, pred = P["y_true"].to_numpy(), P[cols[which]].to_numpy()
    resid = pred - true
    c1, c2 = st.columns(2)
    show(charts.parity(true, pred), c1)
    show(charts.decile_bars(true, pred), c2)

    c3, c4 = st.columns([1.35, 1])
    data = P.assign(residual=resid)
    event = show(charts.residual_map(data, "residual", tiles=False, hover={"y_true": ":.2f", cols[which]: ":.2f"},
                                     title="Residuals: select an area with the box or lasso tool"),
                 c3, key=f"diag_map_{cols[which]}", select=True)
    picked = [i for i in selected_indices(event) if 0 <= i < len(P)]
    with c4:
        if picked:
            sub = data.iloc[picked]
            signs([("Selected block groups", f"{len(picked):,}", f"of {len(P):,} in the test set"),
                   ("MAE in the selection", f"&#36;{np.mean(np.abs(sub['residual'])) * DOLLARS:,.0f}",
                    f"overall &#36;{mae(true, pred) * DOLLARS:,.0f}"),
                   ("Mean signed error", f"&#36;{sub['residual'].mean() * DOLLARS:,.0f}", "positive = over-predicted")])
            table(sub[["y_true", cols[which], "residual", "MedInc", "HouseAge", "AveOccup", "Latitude", "Longitude"]].round(3).head(25))
        else:
            st.info("Use the box or lasso tool in the map's toolbar to select an area and see how the model does there.")
        at_cap = true >= 5.0
        cap = pd.DataFrame({"group": ["at the $500k ceiling", "below the ceiling"],
                            "block groups": [int(at_cap.sum()), int((~at_cap).sum())],
                            "MAE": [usd(np.mean(np.abs(resid[m])) * DOLLARS) if m.any() else "n/a" for m in (at_cap, ~at_cap)],
                            "mean signed error": [usd(np.mean(resid[m]) * DOLLARS) if m.any() else "n/a" for m in (at_cap, ~at_cap)]})
        st.markdown("##### The price ceiling")
        table(cap)
        st.caption(f"A capped block group is only known to be worth at least \\$500,000, so its error is measured against a "
                   f"floor. {int((pred > 5.0).sum())} predictions exceed the ceiling.")
    if which == "Final model":
        k1, k2 = st.columns([1, 1.2])
        imp = pd.DataFrame(b.results["importance"]).rename(columns={"index": "feature"})
        show(charts.importance_bars(imp), k1)
        with k2:
            st.markdown("##### The ten largest test errors of the final model")
            table(pd.DataFrame(b.results["largest_errors"]).drop(columns=["index"], errors="ignore").round(3))
    else:
        st.caption("Permutation importance and the largest-error table were computed in the notebook for the final model only.")
