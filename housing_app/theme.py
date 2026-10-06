"""Design tokens shared by the page styling and every chart (Caltrans sign green, Pacific blue, poppy)."""
from __future__ import annotations

import numpy as np

INK, MUTED, PAPER, FOG = "#1E2B33", "#5D6B73", "#F6F8F7", "#DCE3E1"
SIGN, PACIFIC, POPPY, REDWOOD, GOLD, SAGE, GREY = "#006747", "#1D4E89", "#E8772E", "#A23B2A", "#C9961A", "#6E9B6B", "#9AA6AE"
ACT_C = {"relu": SIGN, "tanh": PACIFIC, "sigmoid": POPPY, "elu": SAGE, "gelu": GOLD}
WIDTH_C = {16: "#8DB8E0", 32: "#6FA3D6", 64: "#3A75B0", 128: "#235A93", 256: "#0E3563", 512: "#071E3D"}
SPLIT_C = {"train": PACIFIC, "validation": POPPY, "test": SIGN}
MODEL_C = {"Final model": SIGN, "Tuned champion": POPPY, "Linear regression": GREY, "Gradient-boosted trees": PACIFIC}
FONT = "Barlow, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
DOLLARS = 100_000
CITIES = {"San Francisco": (37.77, -122.42), "San Jose": (37.34, -121.89), "Sacramento": (38.58, -121.49),
          "Fresno": (36.74, -119.79), "Los Angeles": (34.05, -118.24), "San Diego": (32.72, -117.16)}
GEO_RATIO = float(1 / np.cos(np.deg2rad(37)))

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, label { font-family: __FONT__; }
h1, h2, h3, h4 { font-family: 'Barlow Condensed', __FONT__; font-weight: 600; color: __INK__; }
.block-container { padding-top: 1.2rem; max-width: 1320px; }
.lede { color: __MUTED__; font-size: 1.02rem; max-width: 78ch; margin-top: -4px; }
.sign { background: __SIGN__; color: #fff; border-radius: 10px; border: 3px solid #fff; box-shadow: 0 0 0 2px __SIGN__;
        padding: 14px 22px 13px; }
.sign h1 { color: #fff; font-size: 2.1rem; line-height: 1.08; margin: 0; padding: 0; }
.sign p { color: #E3F1EA; margin: 7px 0 0; font-size: 1rem; max-width: 76ch; }
.signs { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 16px 0 8px; }
.signs .sign { padding: 9px 16px 10px; }
.sign .label { font-size: .9rem; color: #D5ECDF; }
.sign .value { font-family: 'Barlow Condensed', __FONT__; font-size: 2.05rem; font-weight: 700; line-height: 1.05; }
.sign .sub { font-size: .82rem; color: #CBE5D7; }
.trail { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 6px 0 10px; }
.trail .stop { border-left: 4px solid __SIGN__; background: #fff; padding: 8px 12px; border-radius: 0 6px 6px 0; }
.trail .stop b { font-family: 'Barlow Condensed', __FONT__; font-size: 1.2rem; display: block; }
.trail .stop span { color: __MUTED__; font-size: .88rem; }
.brand { font-family: 'Barlow Condensed', __FONT__; font-size: 1.45rem; font-weight: 700; color: __SIGN__; line-height: 1.1; }
.brand-sub { color: __MUTED__; font-size: .9rem; margin-bottom: 10px; }
.muted { color: __MUTED__; font-size: .92rem; }
</style>
"""
CSS = (_CSS.replace("__FONT__", FONT).replace("__INK__", INK).replace("__SIGN__", SIGN).replace("__MUTED__", MUTED))


def register_plotly_template() -> None:
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates["ca_housing"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, color=INK, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PACIFIC, POPPY, SIGN, REDWOOD, GOLD, SAGE],
        xaxis=dict(gridcolor=FOG, zerolinecolor=FOG, linecolor=GREY),
        yaxis=dict(gridcolor=FOG, zerolinecolor=FOG, linecolor=GREY),
        margin=dict(l=10, r=10, t=56, b=10), hoverlabel=dict(font_family=FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0)))
    pio.templates.default = "plotly_white+ca_housing"
