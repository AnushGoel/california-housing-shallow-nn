"""Plotly figure builders: data in, figure out. No Streamlit calls, so each one is easy to test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .theme import CITIES, DOLLARS, GEO_RATIO, GOLD, GREY, INK, MODEL_C, PACIFIC, POPPY, REDWOOD, SIGN

CA_CENTER = {"lat": 37.2, "lon": -119.5}
HAS_TILE_MAPS = hasattr(px, "scatter_map")        # Plotly 5.24+ (MapLibre); older versions use Mapbox traces


def flat_geo(fig, height=540, cities=True):
    fig.update_yaxes(scaleanchor="x", scaleratio=GEO_RATIO, title="Latitude", showgrid=False)
    fig.update_xaxes(title="Longitude", showgrid=False)
    fig.update_layout(height=height)
    if cities:
        for city, (lat, lon) in CITIES.items():
            fig.add_annotation(x=lon, y=lat, text=city, showarrow=False, xanchor="left", xshift=6, yshift=8,
                               font=dict(size=11, color=INK))
    return fig


def tile_map(df, color=None, *, range_color=None, scale="Viridis", hover=None, labels=None, size=5, height=540,
             title="", opacity=0.85, zoom=4.7, center=None):
    kwargs = dict(lat="Latitude", lon="Longitude", color=color, hover_data=hover, labels=labels or {}, opacity=opacity,
                  zoom=zoom, center=center or CA_CENTER, height=height)
    if color is not None and range_color is not None:
        kwargs.update(range_color=range_color, color_continuous_scale=scale)
    if HAS_TILE_MAPS:
        fig = px.scatter_map(df, map_style="carto-positron", **kwargs)
    else:
        fig = px.scatter_mapbox(df, mapbox_style="carto-positron", **kwargs)
    fig.update_traces(marker=dict(size=size))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def value_map(df, var, range_color, tiles=True, title=""):
    data = df.sort_values(var)
    if tiles:
        return tile_map(data, var, range_color=range_color, size=4, title=title)
    fig = px.scatter(data, x="Longitude", y="Latitude", color=var, color_continuous_scale="Viridis",
                     range_color=range_color, render_mode="webgl", opacity=0.85)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title=title)
    return flat_geo(fig)


def residual_map(df, col, *, tiles=False, title="", hover=None, height=540):
    lim = float(np.quantile(np.abs(df[col]), 0.98)) or 1.0
    labels = {col: "pred − true ($100k)"}
    if tiles:
        return tile_map(df, col, range_color=(-lim, lim), scale="RdBu_r", hover=hover, labels=labels, size=5,
                        height=height, title=title)
    fig = px.scatter(df, x="Longitude", y="Latitude", color=col, color_continuous_scale="RdBu_r", range_color=(-lim, lim),
                     render_mode="webgl", hover_data=hover, labels=labels)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(title=title, dragmode="lasso")
    return flat_geo(fig, height=height)


def pin_map(lat, lon, context=None, tiles=True, height=420, zoom=6.2):
    if tiles:
        fig = tile_map(pd.DataFrame({"Latitude": [lat], "Longitude": [lon]}), size=20, height=height, zoom=zoom,
                       center={"lat": lat, "lon": lon})
        fig.update_traces(marker=dict(size=20, color=POPPY))
        return fig
    fig = go.Figure()
    if context is not None:
        fig.add_trace(go.Scattergl(x=context["Longitude"], y=context["Latitude"], mode="markers",
                                   marker=dict(size=3, color="#C9D1D9"), hoverinfo="skip", name="block groups"))
    fig.add_trace(go.Scatter(x=[lon], y=[lat], mode="markers", name="this block group",
                             marker=dict(size=18, symbol="star", color=POPPY, line=dict(color=INK, width=1))))
    fig.update_layout(showlegend=False)
    return flat_geo(fig, height=height)


def _keys(r):
    return ("loss" if r["monitor"] == "val_loss" else "mse"), r["monitor"]


def learning_curves(runs, key, palette, suffix, seed, show_train=True, log_y=True, every_seed=False):
    fig, gap_fig = go.Figure(), go.Figure()
    for _, r in runs:
        level, main = r["cfg"][key], r["cfg"]["seed"] == seed
        if not main and not every_seed:
            continue
        tr_k, va_k = _keys(r)
        ep = list(range(1, len(r["history"][va_k]) + 1))
        color, label = palette.get(level, PACIFIC), f"{level}{suffix}"
        width, alpha = (2.6, 1.0) if main else (1.0, 0.35)
        fig.add_trace(go.Scatter(x=ep, y=r["history"][va_k], mode="lines", line=dict(color=color, width=width),
                                 opacity=alpha, name=f"{label}, validation", legendgroup=label, showlegend=main))
        if show_train:
            fig.add_trace(go.Scatter(x=ep, y=r["history"][tr_k], mode="lines", line=dict(color=color, width=width, dash="dot"),
                                     opacity=alpha, name=f"{label}, training", legendgroup=label, showlegend=main))
        if main:
            fig.add_trace(go.Scatter(x=[r["best_epoch"]], y=[r["best_val_loss"]], mode="markers", legendgroup=label,
                                     marker=dict(symbol="star", size=14, color=color, line=dict(color=INK, width=1)),
                                     showlegend=False, name=f"{label}, best epoch",
                                     hovertemplate=f"{label}<br>best validation MSE %{{y:.4f}}<br>epoch %{{x}}<extra></extra>"))
            gap = np.asarray(r["history"][va_k]) - np.asarray(r["history"][tr_k])
            gap_fig.add_trace(go.Scatter(x=ep, y=gap, mode="lines", line=dict(color=color, width=2.2), name=label))
    fig.update_yaxes(type="log" if log_y else "linear", title="MSE")
    fig.update_layout(title=f"Learning curves, seed {seed} in bold (dotted = training, star = best epoch)", height=470,
                      xaxis_title="Epoch")
    gap_fig.add_hline(y=0, line=dict(color=GREY, dash="dot"))
    gap_fig.update_layout(title="Generalisation gap (validation − training MSE)", height=300, xaxis_title="Epoch",
                          yaxis_title="MSE gap")
    return fig, gap_fig


def animated_learning(runs, key, palette, suffix, seed, step=2):
    mains = [r for _, r in runs if r["cfg"]["seed"] == seed]
    if not mains:
        return None
    max_ep = max(len(r["history"][r["monitor"]]) for r in mains)
    frames = list(range(step, max_ep + 1, step))
    if not frames or frames[-1] != max_ep:
        frames.append(max_ep)
    rows = []
    for r in mains:
        tr_k, va_k = _keys(r)
        label, va, tr = f"{r['cfg'][key]}{suffix}", r["history"][va_k], r["history"][tr_k]
        for f in frames:
            for e in range(min(f, len(va))):
                rows.append((f, e + 1, va[e], label, "validation"))
                rows.append((f, e + 1, tr[e], label, "training"))
    df = pd.DataFrame(rows, columns=["epoch reached", "epoch", "MSE", "model", "curve"])
    late = df.loc[df["epoch"] >= 3, "MSE"]
    top = float(late.quantile(0.99)) if len(late) else float(df["MSE"].max())
    colors = {f"{r['cfg'][key]}{suffix}": palette.get(r["cfg"][key], PACIFIC) for r in mains}
    fig = px.line(df, x="epoch", y="MSE", color="model", line_dash="curve", animation_frame="epoch reached",
                  range_x=[1, max_ep], range_y=[float(df["MSE"].min()) * 0.95, top * 1.08], color_discrete_map=colors,
                  line_dash_map={"validation": "solid", "training": "dot"})
    fig.update_layout(title="Watch the networks learn: press play", height=490, updatemenus=[dict(
        type="buttons", direction="left", showactive=False, x=0, y=1.14, xanchor="left", yanchor="bottom",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, {"frame": {"duration": 60, "redraw": False}, "transition": {"duration": 0}, "fromcurrent": True}]),
                 dict(label="❚❚ Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])])])
    return fig


def seed_strip(runs, key, suffix):
    df = pd.DataFrame([{"level": f"{r['cfg'][key]}{suffix}", "seed": r["cfg"]["seed"], "best validation MSE": r["best_val_loss"]}
                       for _, r in runs])
    fig = px.strip(df, x="level", y="best validation MSE", hover_data={"seed": True}, stripmode="overlay")
    fig.update_traces(marker=dict(size=11, color=PACIFIC, line=dict(color="white", width=1)))
    fig.update_layout(title="Best validation MSE for each seed", height=330, xaxis_title="")
    return fig


def _coded(series):
    levels = sorted(series.unique(), key=lambda v: (isinstance(v, str), v))
    mapping = {v: i for i, v in enumerate(levels)}
    text = [f"{v:g}" if isinstance(v, float) else str(v) for v in levels]
    return series.map(mapping).astype(float).tolist(), list(range(len(levels))), text


def parcoords(trials):
    dims = []
    for col, label in [("hidden_units", "hidden units"), ("activation", "activation"), ("lr", "learning rate"),
                       ("l2", "L2 weight decay"), ("dropout", "dropout"), ("features", "features")]:
        if col == "lr":
            ticks = [3e-4, 1e-3, 3e-3, 1e-2]
            dims.append(dict(label=label, values=np.log10(trials["lr"].astype(float)).tolist(),
                             tickvals=np.log10(ticks).tolist(), ticktext=[f"{t:g}" for t in ticks]))
        else:
            values, tickvals, ticktext = _coded(trials[col])
            dims.append(dict(label=label, values=values, tickvals=tickvals, ticktext=ticktext))
    score = trials["best val MSE"].astype(float).tolist()
    dims.append(dict(label="best validation MSE", values=score))
    fig = go.Figure(go.Parcoords(line=dict(color=score, colorscale="Viridis", reversescale=True, showscale=True,
                                           colorbar=dict(title="val MSE")), dimensions=dims))
    fig.update_layout(title="Every trial as one line: drag along any axis to filter", height=460,
                      margin=dict(l=60, r=40, t=80, b=30))
    return fig


def hp_scatter(trials, hp, ref):
    cmap = {"raw": PACIFIC, "log": POPPY}
    if hp in ("lr", "hidden_units"):
        fig = px.scatter(trials, x=hp, y="best val MSE", color="features", log_x=True, hover_data=["trial", "activation"],
                         color_discrete_map=cmap)
    else:
        fig = px.strip(trials.assign(**{hp: trials[hp].astype(str)}), x=hp, y="best val MSE", color="features",
                       hover_data=["trial"], color_discrete_map=cmap)
    fig.update_traces(marker=dict(size=11))
    fig.add_hline(y=ref, line=dict(color=INK, dash="dot"), annotation_text="Step 5 model", annotation_position="top left")
    fig.update_layout(title=f"Validation MSE by {hp}", height=400)
    return fig


def champion_curves(histories, names):
    fig = go.Figure()
    for (label, run_name), color in zip(names.items(), [PACIFIC, POPPY]):
        r = histories.get(run_name)
        if r is None:
            continue
        tr_k, va_k = _keys(r)
        ep = list(range(1, len(r["history"][va_k]) + 1))
        fig.add_trace(go.Scatter(x=ep, y=r["history"][va_k], name=f"{label}, validation", line=dict(color=color, width=2.4)))
        fig.add_trace(go.Scatter(x=ep, y=r["history"][tr_k], name=f"{label}, training", line=dict(color=color, width=1.4, dash="dot")))
    fig.update_yaxes(type="log", title="MSE")
    fig.update_layout(title="Learning curves: required protocol against the tuned champion", height=400, xaxis_title="Epoch")
    return fig


def forest(df, label, est, lo, hi, *, title, xaxis_title, colors=None, zero_line=False, money=True):
    fmt = (lambda v: f"${v:,.0f}") if money else (lambda v: f"{v:.3f}")
    fig = go.Figure()
    for _, row in df.iterrows():
        color = (colors or {}).get(row[label], PACIFIC)
        fig.add_trace(go.Scatter(x=[row[lo], row[hi]], y=[row[label], row[label]], mode="lines",
                                 line=dict(color=color, width=8), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[row[est]], y=[row[label]], mode="markers+text", text=[fmt(row[est])],
                                 textposition="top center", showlegend=False,
                                 marker=dict(color=color, size=15, line=dict(color="white", width=2)),
                                 hovertemplate=f"{row[label]}: {fmt(row[est])} (interval {fmt(row[lo])} to {fmt(row[hi])})<extra></extra>"))
    if zero_line:
        fig.add_vline(x=0, line=dict(color=INK, dash="dash"))
    fig.update_layout(title=title, xaxis_title=xaxis_title, height=130 + 72 * len(df), yaxis=dict(autorange="reversed", title=""))
    return fig


def boot_hist(draws, metric="MAE"):
    fig = go.Figure()
    for name, d in draws.items():
        fig.add_trace(go.Histogram(x=(np.asarray(d[metric]) * DOLLARS).tolist(), name=name, opacity=0.55, nbinsx=60,
                                   marker_color=MODEL_C.get(name, PACIFIC)))
    fig.update_layout(barmode="overlay", title=f"Bootstrap distribution of the test {metric}", height=360,
                      xaxis_title=f"{metric} ($)", yaxis_title="resamples")
    return fig


def cv_dots(cv, test_mae):
    names = {"random": "random folds", "spatial": "spatial folds (unseen regions)"}
    df = cv.assign(**{"MAE ($)": cv["MAE"] * DOLLARS, "scheme name": cv["scheme"].map(names), "fold #": cv["fold"] + 1})
    fig = px.strip(df, x="scheme name", y="MAE ($)", color="scheme name", hover_data={"fold #": True, "n_test": True},
                   color_discrete_map={names["random"]: PACIFIC, names["spatial"]: POPPY})
    fig.update_traces(marker=dict(size=15, line=dict(color="white", width=1)))
    for scheme, label in names.items():
        mean = float(df.loc[df["scheme"] == scheme, "MAE ($)"].mean())
        fig.add_annotation(x=label, y=mean, text=f"mean ${mean:,.0f}", showarrow=False, xshift=90)
    fig.add_hline(y=test_mae * DOLLARS, line=dict(color=SIGN, dash="dot"), annotation_text="Step 6 test MAE",
                  annotation_position="bottom left")
    fig.update_layout(title="Held-out MAE per fold", height=420, showlegend=False, xaxis_title="")
    return fig


def fold_map(dataset):
    d = dataset[dataset["spatial_fold"] >= 0]
    d = d.assign(fold="fold " + (d["spatial_fold"].astype(int) + 1).astype(str))
    fig = px.scatter(d, x="Longitude", y="Latitude", color="fold", render_mode="webgl", opacity=0.8)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title="Spatial folds: each colour is held out as a whole")
    return flat_geo(fig, height=480, cities=False)


def slice_bars(tbl, overall, title):
    d = tbl.assign(label=[f"{s} (n = {n:,})" for s, n in zip(tbl["slice"], tbl["n"])])
    fig = go.Figure(go.Bar(x=(d["MAE"] * DOLLARS).tolist(), y=d["label"].tolist(), orientation="h", marker_color=PACIFIC,
                           error_x=dict(type="data", symmetric=False, array=((d["high"] - d["MAE"]) * DOLLARS).tolist(),
                                        arrayminus=((d["MAE"] - d["low"]) * DOLLARS).tolist(), color=INK)))
    fig.add_vline(x=overall * DOLLARS, line=dict(color=SIGN, dash="dot"), annotation_text="overall", annotation_position="top")
    fig.update_layout(title=title, height=150 + 46 * len(d), xaxis_title="test MAE ($)", yaxis=dict(autorange="reversed"))
    return fig


def parity(true, pred):
    fig = px.density_heatmap(x=true, y=pred, nbinsx=60, nbinsy=60, color_continuous_scale="Blues",
                             labels={"x": "true value ($100k)", "y": "predicted value ($100k)"})
    top = float(max(5.4, np.max(pred) + 0.1))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=INK, dash="dash"), name="perfect"))
    fig.add_vline(x=5.0, line=dict(color=REDWOOD, dash="dot"), annotation_text="$500k ceiling")
    fig.update_layout(title=f"Predicted against true: MAE ${np.mean(np.abs(np.asarray(pred) - np.asarray(true))) * DOLLARS:,.0f}",
                      height=460, showlegend=False)
    return fig


def decile_bars(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    dec = pd.qcut(true, 10, labels=False, duplicates="drop")
    g = (pd.DataFrame({"decile": dec, "abs": np.abs(pred - true), "res": pred - true, "true": true})
         .groupby("decile").agg(mae=("abs", "mean"), bias=("res", "mean"), lo=("true", "min"), hi=("true", "max")))
    ranges = [f"{a:.2f}–{b:.2f}" for a, b in zip(g["lo"], g["hi"])]
    fig = go.Figure([go.Bar(x=ranges, y=(g["mae"] * DOLLARS).tolist(), name="mean absolute error", marker_color=PACIFIC),
                     go.Bar(x=ranges, y=(g["bias"] * DOLLARS).tolist(), name="mean signed error", marker_color=POPPY)])
    fig.update_layout(title="Error by decile of the true value (positive = over-prediction)", barmode="group", height=460,
                      yaxis_title="dollars", xaxis_title="true value range ($100k)")
    return fig


def importance_bars(imp):
    fig = px.bar(imp.sort_values("mean rise in test MSE"), x="mean rise in test MSE", y="feature", orientation="h",
                 error_x="SD over 10 shuffles", color_discrete_sequence=[SIGN])
    fig.update_layout(title="Permutation importance (rise in test MSE when a feature is shuffled)", height=340)
    return fig


def contributions_bar(contrib, top=24):
    order = np.argsort(-np.abs(contrib))[:top]
    return go.Figure(go.Bar(x=[f"unit {i}" for i in order], y=(contrib[order] * DOLLARS).tolist(),
                            marker_color=[SIGN if v > 0 else REDWOOD for v in contrib[order]]))


def sensitivity_line(grid, preds, x0, y0, feature):
    fig = go.Figure(go.Scatter(x=list(grid), y=list(preds), mode="lines", line=dict(color=PACIFIC, width=2.6), name="prediction"))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(size=13, color=POPPY), name="current"))
    fig.update_layout(title=f"How the prediction responds to {feature}, everything else held fixed", height=380,
                      xaxis_title=feature, yaxis_title="predicted value ($)", showlegend=False)
    return fig


def hex_deck(data, elevation_scale=50.0, radius_m=8000):
    """3D hexagon columns with pydeck (bundled with Streamlit); returns None if pydeck is unavailable."""
    try:
        import pydeck as pdk
    except ImportError:
        return None
    layer = pdk.Layer("HexagonLayer", data=data[["Longitude", "Latitude", "value"]], get_position=["Longitude", "Latitude"],
                      get_elevation_weight="value", elevation_aggregation="MEAN", get_color_weight="value",
                      color_aggregation="MEAN", radius=radius_m, elevation_scale=elevation_scale, elevation_range=[0, 1500],
                      extruded=True, pickable=True, coverage=0.92,
                      color_range=[[237, 248, 233], [186, 228, 179], [116, 196, 118], [49, 163, 84], [0, 109, 44], [0, 68, 27]])
    view = pdk.ViewState(latitude=36.4, longitude=-119.6, zoom=5.1, pitch=50, bearing=-15)
    return pdk.Deck(layers=[layer], initial_view_state=view, map_style=None,
                    tooltip={"text": "average in this hexagon: {elevationValue}"})


def density_tiles(data, value_col="value", title=""):
    kwargs = dict(lat="Latitude", lon="Longitude", z=value_col, radius=12, zoom=4.7, center=CA_CENTER, height=560)
    if hasattr(px, "density_map"):
        fig = px.density_map(data, map_style="carto-positron", **kwargs)
    else:
        fig = px.density_mapbox(data, mapbox_style="carto-positron", **kwargs)
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def waterfall(explanation):
    contribs = sorted(explanation["contributions"], key=lambda c: -abs(c["contribution"]))
    labels = ["typical block group"] + [c["feature"] for c in contribs] + ["this prediction"]
    measure = ["absolute"] + ["relative"] * len(contribs) + ["total"]
    values = [explanation["baseline"]] + [c["contribution"] for c in contribs] + [0.0]
    text = [f"${explanation['baseline']:,.0f}"] + [f"{c['contribution']:+,.0f}" for c in contribs] + [f"${explanation['prediction']:,.0f}"]
    fig = go.Figure(go.Waterfall(x=labels, y=values, measure=measure, text=text, textposition="outside",
                                 connector=dict(line=dict(color=GREY, width=1)), increasing=dict(marker=dict(color=SIGN)),
                                 decreasing=dict(marker=dict(color=REDWOOD)), totals=dict(marker=dict(color=PACIFIC))))
    fig.update_layout(title="Why this price: exact Shapley contributions against a typical block group", height=430,
                      yaxis_title="predicted value ($)", showlegend=False)
    return fig


def psi_bars(drift):
    colors = {"stable": SIGN, "moderate shift": GOLD, "major shift": REDWOOD}
    fig = go.Figure(go.Bar(x=drift["PSI"].tolist(), y=drift["feature"].tolist(), orientation="h",
                           marker_color=[colors.get(s, GREY) for s in drift["status"]],
                           text=[f"{v:.2f}" for v in drift["PSI"]], textposition="outside"))
    for x in (0.10, 0.25):
        fig.add_vline(x=x, line=dict(color=GREY, dash="dot"), annotation_text=f"{x:.2f}", annotation_position="top")
    fig.update_layout(title="Population stability index by feature (higher = further from the training data)", height=380,
                      xaxis_title="PSI", yaxis=dict(autorange="reversed"))
    return fig


def coverage_curve(table, groups=()):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.45, 1.0], y=[0.45, 1.0], mode="lines", line=dict(color=GREY, dash="dash"), name="promise kept exactly"))
    fig.add_trace(go.Scatter(x=table["promised"].tolist(), y=table["achieved"].tolist(), mode="lines+markers",
                             line=dict(color=SIGN, width=3), name="all test block groups"))
    for g, color in zip(groups, [REDWOOD, PACIFIC, POPPY, GOLD]):
        fig.add_trace(go.Scatter(x=table["promised"].tolist(), y=table[g].tolist(), mode="lines+markers",
                                 line=dict(color=color, width=1.6), name=str(g)))
    fig.update_layout(title="Promised against achieved coverage on the test set", height=420, xaxis_title="promised coverage",
                      yaxis_title="achieved coverage", xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"))
    return fig


def prediction_hist(pred_dollars, reference_dollars):
    fig = go.Figure([go.Histogram(x=list(reference_dollars), name="training targets", histnorm="probability density",
                                  opacity=0.45, marker_color=GREY, nbinsx=60),
                     go.Histogram(x=list(pred_dollars), name="predictions for this batch", histnorm="probability density",
                                  opacity=0.7, marker_color=PACIFIC, nbinsx=60)])
    fig.update_layout(barmode="overlay", title="Predicted values against the training distribution", height=380,
                      xaxis_title="median house value ($)")
    return fig


def ood_map(scored):
    fig = px.scatter(scored, x="Longitude", y="Latitude", color="ood_distance", color_continuous_scale="Viridis",
                     render_mode="webgl", hover_data={"prediction": ":,.0f", "out_of_distribution": True},
                     labels={"ood_distance": "distance from training data"})
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(title="Where the batch sits relative to the training data")
    return flat_geo(fig, height=480)
