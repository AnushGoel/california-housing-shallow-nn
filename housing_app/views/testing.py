"""Model testing: bootstrap intervals, paired comparisons, random vs spatial cross-validation and slice tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts
from ..evaluation import compare, interval, mae, nearest_place, paired_bootstrap, r2, rmse, slice_table
from ..uncertainty import coverage_table
from ..theme import CITIES, DOLLARS, GREY, MODEL_C, REDWOOD, SIGN
from .common import page_title, show, signs, table, usd_html

SLICERS = ["nearest major city", "income band", "house age band", "price ceiling", "input extremity"]


@st.cache_data(show_spinner="Resampling the test set…")
def _bootstrap(y, preds: tuple, n_boot: int, seed: int):
    return paired_bootstrap(y, dict(preds), n_boot=n_boot, seed=seed)


def _slice_labels(P, by, net):
    if by == "nearest major city":
        return nearest_place(P["Latitude"], P["Longitude"], CITIES)
    if by == "income band":
        return pd.qcut(P["MedInc"], 5, labels=["lowest 20%", "20-40%", "40-60%", "60-80%", "top 20%"]).astype(str).to_numpy()
    if by == "house age band":
        return pd.cut(P["HouseAge"], [0, 15, 30, 45, 100],
                      labels=["up to 15 years", "16-30 years", "31-45 years", "over 45 years"]).astype(str).to_numpy()
    if by == "price ceiling":
        return np.where(P["y_true"] >= 5.0, "at the $500k ceiling", "below the ceiling")
    z = np.abs(net.inputs(P[list(net.feature_names)].to_numpy())).max(axis=1)
    return np.where(z > 4, "an input beyond 4 SD", "all inputs within 4 SD")


def _fmt_ci(est, lo, hi, money=True):
    if money:
        return f"${est * DOLLARS:,.0f} (${lo * DOLLARS:,.0f} to ${hi * DOLLARS:,.0f})"
    return f"{est:.3f} ({lo:.3f} to {hi:.3f})"


def render(ctx) -> None:
    b = ctx.bundle
    P, cols = b.predictions, b.prediction_columns()
    y = P["y_true"].to_numpy()
    page_title("Testing the model", "A single test score hides its own uncertainty. These tests put error bars on it, check "
               "whether differences between models are real, ask how the network copes with regions it has never seen, and "
               "find where it is weakest.")

    st.markdown("### Bootstrap confidence intervals")
    c1, c2, c3 = st.columns([2, 1, 1])
    chosen = c1.multiselect("Models", list(cols), default=list(cols), key="t_models")
    level = c2.select_slider("Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
                             format_func=lambda v: f"{v:.0%}", key="t_level")
    n_boot = c3.select_slider("Resamples", options=[200, 500, 1000, 2000, 5000], value=1000, key="t_nboot")
    if chosen:
        draws = _bootstrap(y, tuple((m, P[cols[m]].to_numpy()) for m in chosen), int(n_boot), 0)
        rows = []
        for m in chosen:
            p = P[cols[m]].to_numpy()
            row = {"model": m}
            for metric, fn in (("MAE", mae), ("RMSE", rmse), ("R2", r2)):
                lo, hi = interval(draws[m][metric], level)
                row.update({metric: fn(y, p), f"{metric} low": lo, f"{metric} high": hi})
            rows.append(row)
        ci = pd.DataFrame(rows)
        money = ci.assign(**{k: ci[k] * DOLLARS for k in ("MAE", "MAE low", "MAE high")}).sort_values("MAE")
        f1, f2 = st.columns([1.1, 1])
        show(charts.forest(money, "model", "MAE", "MAE low", "MAE high", title=f"Test MAE with {level:.0%} bootstrap intervals",
                           xaxis_title="mean absolute error ($)", colors=MODEL_C), f1)
        show(charts.boot_hist({m: draws[m] for m in chosen}), f2)
        table(pd.DataFrame({"model": ci["model"],
                            "MAE (interval)": [_fmt_ci(*v) for v in zip(ci["MAE"], ci["MAE low"], ci["MAE high"])],
                            "RMSE (interval)": [_fmt_ci(*v) for v in zip(ci["RMSE"], ci["RMSE low"], ci["RMSE high"])],
                            "R² (interval)": [_fmt_ci(*v, money=False) for v in zip(ci["R2"], ci["R2 low"], ci["R2 high"])]}))
        if "Final model" in chosen and len(chosen) > 1:
            st.markdown("#### Paired comparisons with the final model")
            pr = []
            for m in chosen:
                if m == "Final model":
                    continue
                res = compare(draws["Final model"], draws[m], "MAE", level)
                verdict = {"first better": "final model better", "second better": f"{m} better"}.get(res["verdict"], res["verdict"])
                pr.append({"comparison": f"vs {m}", "MAE difference": (mae(y, P[cols["Final model"]]) - mae(y, P[cols[m]])) * DOLLARS, "low": res["low"] * DOLLARS,
                           "high": res["high"] * DOLLARS, "p (two-sided)": res["p_two_sided"], "verdict": verdict})
            pr = pd.DataFrame(pr)
            colors = {r["comparison"]: SIGN if r["verdict"] == "final model better" else
                      (REDWOOD if r["verdict"].endswith("better") else GREY) for _, r in pr.iterrows()}
            show(charts.forest(pr, "comparison", "MAE difference", "low", "high", colors=colors, zero_line=True,
                               title="Final model MAE minus each alternative (negative = final model better)",
                               xaxis_title="difference in MAE ($)"))
            table(pr.round({"MAE difference": 0, "low": 0, "high": 0, "p (two-sided)": 4}))
            st.caption("Every model is scored on the same resampled block groups, so each difference gets its own interval. "
                       "If the interval excludes zero, the gap is unlikely to be an accident of this test sample.")
    else:
        st.info("Pick at least one model to compute intervals.")

    st.markdown("### Cross-validation: random folds against unseen regions")
    ev = b.results.get("evaluation")
    if not ev:
        st.info("These artifacts predate the cross-validation tests. Re-run the notebook (Section 6c) to add them.")
    else:
        cv = pd.DataFrame(ev["cv"])
        means = cv.groupby("scheme")["MAE"].mean() * DOLLARS
        signs([("Random-fold MAE", usd_html(means.get("random", float("nan"))), f"{ev['cv_folds']} folds of the development rows"),
               ("Spatial-fold MAE", usd_html(means.get("spatial", float("nan"))), "whole regions held out together"),
               ("Cost of unseen regions", f"{ev['cv_optimism_pct']:+.1f}%", "spatial relative to random folds")])
        k1, k2 = st.columns([1, 1.2])
        show(charts.cv_dots(cv, b.final["test_mae"]), k1)
        if "spatial_fold" in b.dataset.columns:
            show(charts.fold_map(b.dataset), k2)
        st.caption(f"Each fold retrains the final configuration for {ev.get('cv_epochs', '?')} epochs with the scaler refitted on "
                   f"that fold's training rows. Spatial folds hold out whole regions out of {ev.get('n_regions', '?')} k-means "
                   "clusters, so the gap between the two schemes measures how optimistic a random split is for new places.")

    st.markdown("### Slice tests")
    s1, s2 = st.columns(2)
    model = s1.selectbox("Model", list(cols), key="slice_model")
    by = s2.selectbox("Slice the test set by", SLICERS, key="slice_by")
    p = P[cols[model]].to_numpy()
    tbl = slice_table(y, p, _slice_labels(P, by, b.final_model), n_boot=500, seed=0)
    show(charts.slice_bars(tbl, mae(y, p), title=f"{model}: test MAE by {by} (95% bootstrap intervals)"))
    worst = tbl.iloc[0]
    st.markdown(f"Weakest slice: **{worst['slice']}**, with a mean absolute error of \\${worst['MAE'] * DOLLARS:,.0f} on "
                f"{int(worst['n']):,} block groups, against \\${mae(y, p) * DOLLARS:,.0f} overall.")

    st.markdown("### Conformal prediction intervals")
    net = b.final_model
    if not net.has_intervals:
        st.info("These artifacts carry no calibration residuals. Re-run the notebook (Section 6c.4) to add conformal intervals.")
        return
    g1, g2 = st.columns([1.15, 1])
    group_by = g1.radio("Break coverage down by", ["price ceiling", "nearest major city", "income band"], horizontal=True,
                        key="conf_groups")
    tbl = coverage_table(y, P["y_pred_final"].to_numpy(), net.cal_residuals, groups=_slice_labels(P, group_by, net))
    groups = [c for c in tbl.columns if c not in ("promised", "half_width", "achieved")]
    show(charts.coverage_curve(tbl, groups[:4]), g1)
    level = g2.select_slider("Coverage to inspect", options=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95], value=0.9,
                             format_func=lambda v: f"{v:.0%}", key="conf_level")
    row = tbl[np.isclose(tbl["promised"], level)].iloc[0]
    signs([("Interval half-width", f"&#36;{row['half_width'] * DOLLARS:,.0f}", f"for {level:.0%} coverage"),
           ("Achieved on the test set", f"{row['achieved']:.1%}", f"promised {level:.0%}")], g2)
    table(pd.DataFrame({"group": groups, f"coverage at {level:.0%}": [f"{row[g]:.1%}" for g in groups]}), g2)
    st.caption("Split conformal prediction turns the validation residuals into a distribution-free interval (Vovk et al., "
               "2005; Lei et al., 2018). The guarantee is marginal: it holds on average over block groups, not for every "
               "group, which is why coverage at the price ceiling can fall short.")
