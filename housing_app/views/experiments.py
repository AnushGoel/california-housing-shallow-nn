"""Steps 4 and 5 interactively: learning curves, an animation, every seed and the generalisation gap."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts
from ..theme import ACT_C, WIDTH_C
from .common import page_title, show, table


def required_runs(bundle, **conditions):
    """Runs trained with the brief's protocol (Adam defaults), filtered by configuration values."""
    out = []
    for name, r in bundle.histories.items():
        cfg = r["cfg"]
        if cfg.get("lr") is None and not name.startswith("baseline") and all(cfg.get(k) == v for k, v in conditions.items()):
            out.append((name, r))
    return out


def render(ctx) -> None:
    b = ctx.bundle
    seed = b.meta["seed"]
    page_title("Experiments", "Steps 4 and 5 changed one thing at a time and kept everything else fixed. Each "
               f"configuration was also repeated on {len(b.meta.get('robustness_seeds', [seed]))} seeds to separate real "
               "effects from training noise.")
    view = st.radio("Experiment", ["Hidden-layer size (Step 4)", "Activation function (Step 5)"], horizontal=True, key="exp_view")
    o1, o2, o3 = st.columns(3)
    show_train = o1.checkbox("Show training curves", value=True, key="exp_train")
    log_y = o2.checkbox("Logarithmic y axis", value=True, key="exp_log")
    every_seed = o3.checkbox("Show every seed", value=False, key="exp_seeds")
    if view.startswith("Hidden"):
        runs, key, palette, suffix = required_runs(b, activation="relu"), "hidden_units", WIDTH_C, " units"
        section = b.results["widths"]
    else:
        runs, key, palette, suffix = required_runs(b, hidden_units=b.results["widths"]["selected"]), "activation", ACT_C, ""
        section = b.results["activations"]
    runs = sorted(runs, key=lambda nr: (str(nr[1]["cfg"][key]).zfill(4), nr[1]["cfg"]["seed"]))
    if not runs:
        st.info("No learning curves were found for this experiment in the artifacts.")
        return
    fig, gap_fig = charts.learning_curves(runs, key, palette, suffix, seed, show_train, log_y, every_seed)
    show(fig)
    with st.expander("Watch the networks learn (animation)", expanded=False):
        anim = charts.animated_learning(runs, key, palette, suffix, seed)
        if anim is not None:
            show(anim)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(f"##### Results, seed {seed}")
        table(pd.DataFrame(section["table"]).round(4))
        st.markdown("##### Across seeds")
        table(pd.DataFrame(section["seed_table"]).round(4))
        decision = section.get("decision")
        if decision:
            table(pd.DataFrame({"decision step": list(decision), "result": list(decision.values())}))
    show(charts.seed_strip(runs, key, suffix), c2)
    show(gap_fig, c2)
