"""Model-testing toolkit: metrics, paired percentile bootstrap, model comparisons and slice tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def paired_bootstrap(y, preds: dict, n_boot: int = 1000, seed: int = 0, chunk: int = 250) -> dict:
    """Score every model on the same resamples of the rows. Returns name -> {'MAE','RMSE','R2'} arrays."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    out = {name: {"MAE": [], "RMSE": [], "R2": []} for name in preds}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, len(y), size=(b, len(y)))
        yb = y[idx]
        sst = ((yb - yb.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        sst = np.where(sst > 0, sst, np.nan)
        for name, p in preds.items():
            err = np.asarray(p, dtype=float)[idx] - yb
            out[name]["MAE"].append(np.abs(err).mean(axis=1))
            out[name]["RMSE"].append(np.sqrt((err ** 2).mean(axis=1)))
            out[name]["R2"].append(1 - (err ** 2).sum(axis=1) / sst)
        done += b
    return {name: {k: np.concatenate(v) for k, v in d.items()} for name, d in out.items()}


def interval(samples, level: float = 0.95):
    tail = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(np.asarray(samples, float), [tail, 100 - tail])
    return float(lo), float(hi)


def compare(draws_a: dict, draws_b: dict, metric: str = "MAE", level: float = 0.95) -> dict:
    """Paired comparison of model A against model B on shared resamples (A minus B)."""
    d = np.asarray(draws_a[metric], float) - np.asarray(draws_b[metric], float)
    lo, hi = interval(d, level)
    share = float(np.mean(d > 0) + 0.5 * np.mean(d == 0))
    lower_is_better = metric != "R2"
    if hi < 0:
        verdict = "first better" if lower_is_better else "second better"
    elif lo > 0:
        verdict = "second better" if lower_is_better else "first better"
    else:
        verdict = "no clear difference"
    return {"difference": float(np.mean(d)), "low": lo, "high": hi,
            "p_two_sided": max(2 * min(share, 1 - share), 1 / len(d)), "verdict": verdict}


def slice_table(y, p, labels, n_boot: int = 500, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """MAE per slice with a bootstrap interval, worst slice first."""
    y, p, labels = np.asarray(y, float), np.asarray(p, float), np.asarray(labels)
    rng = np.random.default_rng(seed)
    rows = []
    for lab in pd.unique(labels):
        m = labels == lab
        n = int(m.sum())
        ae = np.abs(p[m] - y[m])
        lo, hi = interval(ae[rng.integers(0, n, size=(n_boot, n))].mean(axis=1), level)
        rows.append({"slice": str(lab), "n": n, "MAE": float(ae.mean()), "low": lo, "high": hi,
                     "bias": float((p[m] - y[m]).mean())})
    return pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)


def nearest_place(lat, lon, places: dict) -> np.ndarray:
    names = list(places)
    c = np.array([places[n] for n in names], dtype=float)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    d = (lat[:, None] - c[None, :, 0]) ** 2 + ((lon[:, None] - c[None, :, 1]) * np.cos(np.deg2rad(37))) ** 2
    return np.array(names)[d.argmin(axis=1)]
