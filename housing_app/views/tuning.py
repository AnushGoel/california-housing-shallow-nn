"""Extension 5b: the random search, as parallel coordinates, scatter plots and learning curves."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts
from .common import page_title, show, table


def render(ctx) -> None:
    b = ctx.bundle
    T = pd.DataFrame(b.results["hpo"]["trials"])
    champ = int(b.results["hpo"]["champion_trial"])
    confirm_seeds = b.meta.get("hpo_confirm_seeds", [])
    page_title("Tuning", "A random search over one-hidden-layer networks, scored on validation MSE only. Trial 0 re-trains "
               f"the Step 5 configuration under the tuning protocol. The champion (trial {champ}) has the lowest mean "
               f"validation MSE after the top three were re-trained on {len(confirm_seeds)} seeds.")
    show(charts.parcoords(T))
    c1, c2 = st.columns([1, 1.2])
    hp = c1.selectbox("Compare validation MSE against", ["lr", "hidden_units", "activation", "l2", "dropout", "features"],
                      key="hp_choice")
    show(charts.hp_scatter(T, hp, b.final["best_val_loss"]), c1)
    names = {"Step 5 model": f"w{b.final['units']}_{b.final['activation']}_s{b.meta['seed']}", f"Champion, trial {champ}": f"hpo{champ:02d}"}
    show(charts.champion_curves(b.histories, names), c2)
    st.markdown("##### All trials")
    table(T.sort_values("best val MSE").round({"lr": 5, "best val MSE": 4, "minutes": 2}))
    st.markdown("##### Top three, re-trained on more seeds")
    table(pd.DataFrame(b.results["hpo"].get("confirm_table", [])).round(4))
