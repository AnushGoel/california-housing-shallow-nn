"""Explore the data: the price ceiling, maps, a 3D view, single features and correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .. import charts
from ..theme import DOLLARS, PACIFIC, REDWOOD, SPLIT_C
from .common import page_title, show, table


def render(ctx) -> None:
    b = ctx.bundle
    ds, target, features = b.dataset, b.target, b.features
    info = b.results["data"].get("feature_info", {})
    page_title("Explore the data", f"{b.results['data']['n_rows']:,} census block groups from 1990. Each one is a "
               "neighbourhood of roughly 600 to 3,000 people, not a single house.")
    c1, c2 = st.columns([1, 1.25])
    ceiling = np.where(ds[target] >= 5.0, "at the $500k ceiling", "below the ceiling")
    fig = px.histogram(ds.assign(ceiling=ceiling), x=target, color="ceiling", nbins=60,
                       color_discrete_map={"below the ceiling": PACIFIC, "at the $500k ceiling": REDWOOD},
                       labels={target: "median house value ($100k)", "ceiling": ""})
    cap = b.results["data"]
    fig.update_layout(title=f"The target is top-coded: {cap.get('cap_count', 0):,} block groups ({cap.get('cap_share', 0):.1%}) "
                            "sit at the ceiling", height=430, bargap=0.02)
    show(fig, c1)
    with c2:
        var = st.selectbox("Colour the map by", [target] + [f for f in features if f not in ("Latitude", "Longitude")],
                           key="explore_var")
        tiles = st.toggle("Street-map background", value=True, key="explore_tiles")
    lo, hi = np.quantile(ds[var], [0.01, 0.99])
    show(charts.value_map(ds, var, (float(lo), float(hi)), tiles=tiles,
                          title=f"{var} across California (colour clipped to the 1st-99th percentile)"), c2)

    st.markdown("### California in 3D")
    c3, c4 = st.columns([1, 3])
    metric = c3.radio("Column height shows", ["median house value", "final model's error (test set)"], key="hex_metric")
    scale = c3.slider("Height exaggeration", 10.0, 120.0, 50.0, step=5.0, key="hex_scale")
    radius = c3.select_slider("Hexagon width (km)", options=[4, 6, 8, 12, 16], value=8, key="hex_radius")
    if metric.startswith("median"):
        data = ds[["Longitude", "Latitude"]].assign(value=ds[target] * DOLLARS)
    else:
        P = b.predictions
        data = P[["Longitude", "Latitude"]].assign(value=(P["y_pred_final"] - P["y_true"]).abs() * DOLLARS)
    deck = charts.hex_deck(data, elevation_scale=scale, radius_m=radius * 1000)
    if deck is not None:
        c4.pydeck_chart(deck)
    else:
        show(charts.density_tiles(data, "value", title=f"Density of {metric}"), c4)
    c3.caption("Drag to pan, right-drag or Ctrl-drag to tilt and rotate, scroll to zoom. Hover a column to read its average "
               "in dollars.")

    st.markdown("### One feature at a time")
    c5, c6 = st.columns([1, 2.2])
    feat = c5.selectbox("Feature", features, index=features.index("AveOccup") if "AveOccup" in features else 0,
                        key="explore_feat")
    scale_choice = c5.radio("Scale", ["original units", "log(1 + x)"], horizontal=True, key="explore_scale")
    meaning, unit = (info.get(feat) or ["", ""])[:2]
    c5.markdown(f"**{feat}**: {meaning} ({str(unit).replace('$', '&#36;')}).", unsafe_allow_html=True)
    summary = pd.DataFrame(b.results["data"]["summary"]).T
    if feat in summary.index:
        table(summary.loc[[feat]].T.rename(columns={feat: "value"}).round(3), c5, index=True)
    use_log = scale_choice.startswith("log") and feat not in ("Latitude", "Longitude")
    vals = np.log1p(ds[feat]) if use_log else ds[feat]
    upper = float(np.quantile(vals, 0.995))
    fig = px.histogram(ds.assign(value=vals)[vals <= upper], x="value", color="split", barmode="overlay",
                       histnorm="probability density", nbins=90, opacity=0.55, color_discrete_map=SPLIT_C,
                       labels={"value": f"log(1 + {feat})" if use_log else feat})
    fig.update_layout(title=f"{feat} by split (top 0.5% of values cut off for readability)", height=420)
    show(fig, c6)

    corr = ds[features + [target]].corr(method="spearman")
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Spearman rank correlations", height=520)
    show(fig)
