"""Shared Streamlit helpers: layout pieces, version-safe chart display, selections and cached reads."""
from __future__ import annotations

import inspect

import streamlit as st

from ..theme import CSS

_CHART_PARAMS = inspect.signature(st.plotly_chart).parameters
_TABLE_PARAMS = inspect.signature(st.dataframe).parameters


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def usd(v) -> str:
    return f"${v:,.0f}"


def usd_html(v) -> str:
    return f"&#36;{v:,.0f}"            # HTML entity, so Streamlit never mistakes '$' for a maths delimiter


def show(fig, where=None, key=None, select=False):
    """Full-width Plotly chart on any Streamlit version; with select=True returns box/lasso selections."""
    target = st if where is None else where
    kwargs = {"width": "stretch"} if "width" in _CHART_PARAMS else {"use_container_width": True}
    if key is not None:
        kwargs["key"] = key
    if select and "on_select" in _CHART_PARAMS:
        kwargs.update(on_select="rerun", selection_mode=("box", "lasso"))
    return target.plotly_chart(fig, **kwargs)


def table(df, where=None, index=False):
    target = st if where is None else where
    if "hide_index" in _TABLE_PARAMS:
        return target.dataframe(df, hide_index=not index)
    return target.dataframe(df)


def hero(title, text, where=None) -> None:
    (st if where is None else where).markdown(f'<div class="sign"><h1>{title}</h1><p>{text}</p></div>',
                                              unsafe_allow_html=True)


def signs(items, where=None) -> None:
    cells = "".join(f'<div class="sign"><div class="label">{a}</div><div class="value">{b}</div><div class="sub">{c}</div></div>'
                    for a, b, c in items)
    (st if where is None else where).markdown(f'<div class="signs">{cells}</div>', unsafe_allow_html=True)


def page_title(title, text=None) -> None:
    st.markdown(f"## {title}")
    if text:
        st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def fixture_banner(bundle) -> None:
    if bundle.is_test_fixture:
        st.warning("These numbers come from the synthetic test fixtures, not from a real training run. Run the notebook "
                   "and commit its artifacts/ folder before deploying.")


def selected_indices(event) -> list:
    """Row positions picked with the box or lasso tool (empty list if nothing is selected)."""
    if event is None:
        return []
    sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if not sel:
        return []
    get = sel.get if isinstance(sel, dict) else (lambda k, d=None: getattr(sel, k, d))
    idx = get("point_indices", None)
    if idx:
        return [int(i) for i in idx]
    out = []
    for p in get("points", []) or []:
        i = p.get("point_index", p.get("pointIndex")) if isinstance(p, dict) else None
        if i is not None:
            out.append(int(i))
    return out


@st.cache_data(ttl=30, show_spinner=False)
def recent_records(_store, cache_label: str, limit: int = 50) -> list:
    """Newest saved records, cached for 30 s; cache_label keeps different stores apart."""
    return _store.recent(limit)
