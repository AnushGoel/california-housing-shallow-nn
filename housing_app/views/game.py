"""Beat the model: guess the 1990 value of real test-set block groups, then compare with the network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts
from ..evaluation import nearest_place
from ..theme import CITIES, DOLLARS
from .common import page_title, recent_records, show, signs, table, usd, usd_html

ROUNDS = 5


def points(guess: float, truth: float) -> int:
    """100 for a perfect guess, falling linearly to 0 at 100% error."""
    return int(round(100 * max(0.0, 1 - abs(guess - truth) / truth)))


def _new_game(n: int) -> None:
    rows = np.random.default_rng().choice(n, size=min(ROUNDS, n), replace=False)
    st.session_state["game"] = {"round": 1, "rows": [int(i) for i in rows], "revealed": False, "you": 0, "model": 0,
                                "wins": 0, "log": [], "saved": False}


def _next_round() -> None:
    game = st.session_state["game"]
    game["round"] += 1
    game["revealed"] = False


def render(ctx) -> None:
    b = ctx.bundle
    P, n = b.predictions, len(b.predictions)
    if "game" not in st.session_state:
        _new_game(n)
    g = st.session_state["game"]
    page_title("Beat the model", f"{ROUNDS} rounds. Each round shows a real block group from the test set, which the network "
               "never trained on. Guess its 1990 median house value, then see whether you got closer than the model.")
    finished = g["round"] > len(g["rows"])
    if not finished:
        row = P.iloc[g["rows"][g["round"] - 1]]
        truth, model_guess = float(row["y_true"]) * DOLLARS, float(row["y_pred_final"]) * DOLLARS
        c1, c2 = st.columns([1.25, 1])
        tiles = c1.toggle("Street-map background", value=True, key="game_tiles")
        show(charts.pin_map(float(row["Latitude"]), float(row["Longitude"]), context=P, tiles=tiles, height=470, zoom=7.0), c1)
        with c2:
            st.markdown(f"#### Round {g['round']} of {len(g['rows'])}")
            city = nearest_place([row["Latitude"]], [row["Longitude"]], CITIES)[0]
            table(pd.DataFrame({"clue": ["nearest major city", "median household income", "median house age",
                                         "rooms per household", "people per household", "residents"],
                                "value": [city, usd(row["MedInc"] * 10_000), f"{row['HouseAge']:.0f} years",
                                          f"{row['AveRooms']:.1f}", f"{row['AveOccup']:.1f}", f"{row['Population']:,.0f}"]}))
            with st.form("guess_form"):
                guess_k = st.slider("Your guess, in thousands of dollars", 15.0, 500.0, 200.0, step=5.0, key="guess_k")
                locked = st.form_submit_button("Lock in my guess", disabled=g["revealed"])
            if locked and not g["revealed"]:
                guess = guess_k * 1000
                you, model_pts = points(guess, truth), points(model_guess, truth)
                closer = abs(guess - truth) < abs(model_guess - truth)
                g.update(you=g["you"] + you, model=g["model"] + model_pts, wins=g["wins"] + int(closer), revealed=True)
                g["log"].append({"round": g["round"], "guess": guess, "model": model_guess, "truth": truth,
                                 "you_pts": you, "model_pts": model_pts, "closer": "you" if closer else "model"})
            if g["revealed"]:
                last = g["log"][-1]
                signs([("True value", usd_html(last["truth"]), "1990 census"),
                       ("Your guess", usd_html(last["guess"]), f"{last['you_pts']} points"),
                       ("Model's guess", usd_html(last["model"]), f"{last['model_pts']} points")])
                if last["closer"] == "you":
                    st.success("You got closer than the model this round.")
                else:
                    st.info("The model got closer this round.")
                st.button("Next block group" if g["round"] < len(g["rows"]) else "See the final score",
                          on_click=_next_round, key="game_next")
    st.markdown(f"**Score:** you {g['you']} points, the model {g['model']} points. You were closer in {g['wins']} of "
                f"{len(g['log'])} rounds.")
    if g["log"]:
        table(pd.DataFrame([{"round": e["round"], "your guess": usd(e["guess"]), "model's guess": usd(e["model"]),
                             "true value": usd(e["truth"]), "your points": e["you_pts"], "model's points": e["model_pts"],
                             "closer": e["closer"]} for e in g["log"]]))
    if finished:
        verdict = "You beat the model!" if g["you"] > g["model"] else ("A draw." if g["you"] == g["model"] else "The model wins this time.")
        signs([("Your total", f"{g['you']} points", verdict), ("Model's total", f"{g['model']} points", f"{len(g['log'])} rounds")])
        if not g["saved"]:
            with st.form("leaderboard_form"):
                player = st.text_input("Name for the leaderboard", max_chars=24, key="player_name")
                submit = st.form_submit_button("Save my score")
            if submit:
                try:
                    ctx.scores.add({"name": (player or "").strip()[:24] or "Anonymous", "points": g["you"],
                                    "model_points": g["model"], "rounds": len(g["log"]), "rounds_closer": g["wins"]})
                    g["saved"] = True
                    recent_records.clear()
                    st.success(f"Score saved to {ctx.user_storage_label}.")
                except Exception as err:
                    st.error(f"Could not save the score: {err}")
        st.button("Play again", on_click=_new_game, args=(n,), key="game_again")
    st.markdown("#### Leaderboard")
    scores = recent_records(ctx.scores, f"{ctx.user_storage_label}/scores", 300)
    board = pd.DataFrame(scores)
    if not board.empty and "points" in board.columns:
        keep = [c for c in ("name", "points", "model_points", "rounds_closer", "saved_at_utc") if c in board.columns]
        table(board.sort_values("points", ascending=False)[keep].head(10).rename(columns={
            "model_points": "model's points", "rounds_closer": "rounds closer than the model", "saved_at_utc": "saved (UTC)"}))
    else:
        st.caption("No scores yet. Finish a game to be the first on the board.")
