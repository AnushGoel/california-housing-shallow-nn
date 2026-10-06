"""Batch scoring and drift: many block groups at once, with intervals, out-of-distribution flags and drift checks."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts
from ..theme import DOLLARS
from .common import page_title, show, signs, table, usd_html

MAX_ROWS = 50_000
SOURCES = ["A sample of held-out block groups", "A simulated market shift", "Upload a CSV"]
MODEL_KEYS = {"Final model": "final", "Tuned champion": "tuned"}


def render(ctx) -> None:
    b, service, features = ctx.bundle, ctx.service, ctx.bundle.features
    page_title("Batch scoring and drift", "Score many block groups at once. Every row gets a conformal interval and an "
               "out-of-distribution check, and the batch as a whole is compared with the training data, so you can see when "
               "the model is being asked about a world it has not seen.")
    c1, c2, c3 = st.columns([1.5, 1, 1])
    source = c1.radio("Data", SOURCES, key="batch_source")
    key = MODEL_KEYS[c2.radio("Model", list(MODEL_KEYS), key="batch_model")]
    level = c3.select_slider("Interval coverage", options=[0.8, 0.9, 0.95], value=0.9, format_func=lambda v: f"{v:.0%}",
                             key="batch_level")
    pool = b.predictions[features]
    if source == SOURCES[2]:
        upload = st.file_uploader("CSV with the columns " + ", ".join(features), type=["csv"], key="batch_upload")
        st.download_button("Download a template CSV", data=pool.head(5).to_csv(index=False).encode("utf-8"),
                           file_name="block_groups_template.csv", mime="text/csv", key="batch_template")
        if upload is None:
            st.info("Upload a CSV to score it, or choose one of the other data sources.")
            return
        try:
            df = pd.read_csv(upload, nrows=MAX_ROWS)
        except Exception as err:
            st.error(f"Could not read the file as CSV: {err}")
            return
    else:
        n = st.select_slider("Rows", options=[100, 250, 500, 1000, 2000], value=500, key="batch_rows")
        df = pool.sample(min(n, len(pool)), random_state=0).reset_index(drop=True)
        if source == SOURCES[1]:
            s1, s2, s3 = st.columns(3)
            income = s1.slider("Change in median income (%)", -30.0, 60.0, 25.0, step=5.0, key="shift_income")
            crowd = s2.slider("Extra people per household", 0.0, 2.0, 0.5, step=0.1, key="shift_crowd")
            north = s3.slider("Move every block group north (degrees)", -1.0, 1.0, 0.0, step=0.1, key="shift_north")
            df = df.assign(MedInc=df["MedInc"] * (1 + income / 100), AveOccup=df["AveOccup"] + crowd,
                           Latitude=df["Latitude"] + north)
    scored, problems = service.predict_frame(df, key, level)
    drift, _ = service.drift(df)
    for problem in problems:
        st.warning(problem)
    valid = scored[scored["valid"]]
    if valid.empty:
        st.error("No row could be scored.")
        return
    moving = int((drift["status"] != "stable").sum()) if not drift.empty else 0
    signs([("Rows scored", f"{len(valid):,}", f"of {len(scored):,} supplied"),
           ("Mean predicted value", usd_html(valid["prediction"].mean()),
            f"{level:.0%} intervals attached" if "lower" in valid else "no calibration data"),
           ("Outside the training data", f"{valid['out_of_distribution'].mean():.1%}", "beyond the training 99th percentile"),
           ("Features that drifted", f"{moving} of {len(drift)}", "PSI of 0.10 or more")])
    k1, k2 = st.columns(2)
    if not drift.empty:
        show(charts.psi_bars(drift), k1)
    ds = b.dataset
    show(charts.prediction_hist(valid["prediction"], ds.loc[ds["split"] == "train", b.target] * DOLLARS), k2)
    k3, k4 = st.columns([1.3, 1])
    show(charts.ood_map(valid), k3)
    with k4:
        st.markdown("##### Drift by feature")
        if not drift.empty:
            table(drift.round(3))
    st.markdown("##### Scored rows")
    cols = [c for c in features + ["prediction", "lower", "upper", "out_of_distribution", "ood_distance"] if c in scored.columns]
    table(scored[cols].round(3).head(200))
    st.download_button("Download all scored rows (CSV)", data=scored.to_csv(index=False).encode("utf-8"),
                       file_name="scored_block_groups.csv", mime="text/csv", key="batch_download")
    st.caption("A PSI below 0.10 is usually read as stable, 0.10 to 0.25 as a moderate shift and above 0.25 as a major shift "
               "(Siddiqi, 2006). The out-of-distribution flag uses the Mahalanobis distance in the model's input space "
               "(Mahalanobis, 1936), with the threshold set at the training data's 99th percentile.")
