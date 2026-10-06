"""Try a block group: sliders, a live NumPy prediction, the hidden units behind it, and saved scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts
from ..artifacts import MODEL_LABELS
from ..theme import CITIES, DOLLARS

MODEL_KEYS = {label: key for key, label in MODEL_LABELS.items()}
from .common import page_title, recent_records, show, signs, table, usd, usd_html


def _reroll() -> None:
    st.session_state["rand_i"] = st.session_state.get("rand_i", 0) + 1


def render(ctx) -> None:
    b = ctx.bundle
    ds, P, features = b.dataset, b.predictions, b.features
    info = b.results["data"].get("feature_info", {})
    page_title("Try a block group", "Move the sliders and the network re-prices the block group instantly. The app runs the "
               "network's forward pass in NumPy from the exported weights.")
    mname = st.radio("Model", list(b.models), horizontal=True, key="try_model")
    net = b.models[mname]
    med = ds[ds["split"] == "train"][features].median()
    presets = {"Typical block group (training medians)": med}
    for city in ["San Francisco", "Los Angeles", "San Diego", "Sacramento", "Fresno"]:
        v = med.copy()
        v["Latitude"], v["Longitude"] = CITIES[city]
        presets[f"The same typical block group, placed in {city}"] = v
    presets["A random block group from the test set"] = None
    st.session_state.setdefault("rand_i", 0)

    c1, c2 = st.columns([1, 1.3])
    choice = c1.selectbox("Start from", list(presets), key="try_preset")
    true_val = None
    if presets[choice] is None:
        c1.button("Pick another block group", on_click=_reroll, key="try_reroll")
        row = P.sample(1, random_state=int(st.session_state["rand_i"])).iloc[0]
        base, true_val = row[features].astype(float), float(row["y_true"])
    else:
        base = presets[choice].astype(float)
    preset_id = f"{choice}|{st.session_state['rand_i']}"
    if st.session_state.get("preset_id") != preset_id:
        for f in features:
            st.session_state[f"feat_{f}"] = float(base[f])
        st.session_state["preset_id"] = preset_id

    values = {}
    left, right = c1.columns(2)
    for k, f in enumerate(features):
        if f in ("Latitude", "Longitude", "HouseAge"):
            lo_f, hi_f = float(ds[f].min()), float(ds[f].max())
        else:
            lo_f, hi_f = (float(v) for v in np.quantile(ds[f], [0.005, 0.995]))
        lo_f, hi_f = min(lo_f, float(base[f])), max(hi_f, float(base[f]))
        step = 1.0 if f == "HouseAge" else (0.01 if f in ("Latitude", "Longitude") else float(max((hi_f - lo_f) / 400, 1e-3)))
        values[f] = (left if k % 2 == 0 else right).slider(f, min_value=lo_f, max_value=hi_f, step=step, key=f"feat_{f}",
                                                           help=(info.get(f) or [""])[0])
    x = np.array([[values[f] for f in features]], dtype=np.float64)
    pred = float(net.predict(x)[0]) * DOLLARS
    level = c2.select_slider("Interval coverage", options=[0.8, 0.9, 0.95], value=0.9, format_func=lambda v: f"{v:.0%}",
                             key="try_level")
    sub = net.describe()
    if net.has_intervals:
        lo, hi = net.interval(x, level)
        sub = f"{level:.0%} interval {usd_html(max(float(lo[0]), 0.0) * DOLLARS)} to {usd_html(float(hi[0]) * DOLLARS)}"
    cards = [("Predicted median house value", usd_html(pred), sub)]
    if true_val is not None:
        cards.append(("True value (1990 census)", usd_html(true_val * DOLLARS), "this block group is in the test set"))
    signs(cards, c2)
    tiles = c2.toggle("Street-map background", value=True, key="try_tiles")
    show(charts.pin_map(values["Latitude"], values["Longitude"], context=ds.sample(min(5000, len(ds)), random_state=0),
                        tiles=tiles, height=380), c2)

    c3, c4 = st.columns(2)
    Z, A = net.hidden(x)
    active = int((Z[0] > 0).sum()) if net.activation == "relu" else int((np.abs(A[0]) > 1e-3).sum())
    fig = charts.contributions_bar(net.contributions(x))
    fig.update_layout(title=f"Hidden units pushing this prediction up or down ({active} of {net.n_units} active)", height=380,
                      yaxis_title="contribution to the prediction ($)")
    show(fig, c3)
    sens = c4.selectbox("Vary one feature", features, key="try_sens")
    lo_s, hi_s = np.quantile(ds[sens], [0.01, 0.99])
    grid = np.linspace(min(lo_s, values[sens]), max(hi_s, values[sens]), 80)
    X_grid = np.repeat(x, len(grid), axis=0)
    X_grid[:, features.index(sens)] = grid
    show(charts.sensitivity_line(grid, net.predict(X_grid) * DOLLARS, values[sens], pred, sens), c4)

    explanation = ctx.service.explain({f: float(values[f]) for f in features}, MODEL_KEYS[mname])
    show(charts.waterfall(explanation))
    st.caption("Each bar is an exact Shapley value: the average change in the prediction when that feature moves from the "
               "typical block group's value to this one's, over every order of moving the features (Shapley, 1953; "
               "Lundberg & Lee, 2017). Latitude and longitude move together as one player, location.")

    distance = np.sqrt(((net.inputs(ds[features].to_numpy(dtype=np.float64)) - net.inputs(x)) ** 2).sum(axis=1))
    nearest = ds.iloc[np.argsort(distance)[:6]]
    near = nearest[features + ["split"]].assign(**{
        "true value": [usd(v * DOLLARS) for v in nearest[b.target]],
        "predicted": [usd(v * DOLLARS) for v in net.predict(nearest[features].to_numpy(dtype=np.float64))]})
    st.markdown("##### The six most similar real block groups (distance in the model's standardised input space)")
    table(near[["true value", "predicted", "split"] + features].round(2))

    st.markdown("##### Save this scenario")
    with st.form("save_scenario", clear_on_submit=True):
        label = st.text_input("Name", max_chars=40, placeholder="for example: Fresno starter home", key="scenario_name")
        save = st.form_submit_button("Save scenario")
    if save:
        try:
            ctx.scenarios.add({"name": (label or "").strip()[:40] or "Untitled", "model": mname, "prediction_usd": round(pred),
                               "inputs": {f: round(float(values[f]), 4) for f in features}})
            recent_records.clear()
            st.success(f"Saved to {ctx.user_storage_label}.")
        except Exception as err:
            st.error(f"Could not save the scenario: {err}")
    if not ctx.user_storage_is_cloud:
        st.caption("Saves go to this server's disk. Streamlit Community Cloud wipes that disk when the app restarts, so add a "
                   "bucket to the app's secrets to keep them (see the README).")
    recent = recent_records(ctx.scenarios, f"{ctx.user_storage_label}/scenarios", 20)
    if recent:
        table(pd.DataFrame([{"saved (UTC)": r.get("saved_at_utc"), "name": r.get("name"), "model": r.get("model"),
                             "prediction": usd(r.get("prediction_usd", 0)),
                             **{k: r.get("inputs", {}).get(k) for k in ("MedInc", "HouseAge", "Latitude", "Longitude")}}
                            for r in recent]))
