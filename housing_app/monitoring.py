"""Batch-scoring safety checks: input validation, out-of-distribution scores and drift statistics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Physical plausibility, deliberately wider than the training range: values outside these are data errors.
PLAUSIBLE = {"MedInc": (0.0, 30.0), "HouseAge": (0.0, 150.0), "AveRooms": (0.1, 500.0), "AveBedrms": (0.05, 200.0),
             "Population": (1.0, 100_000.0), "AveOccup": (0.1, 5_000.0), "Latitude": (30.0, 45.0),
             "Longitude": (-130.0, -110.0)}
PSI_BANDS = ((0.10, "stable"), (0.25, "moderate shift"), (np.inf, "major shift"))   # common rule of thumb (Siddiqi, 2006)


def validate_frame(df: pd.DataFrame, features: list):
    """Return (numeric feature frame, row-is-usable mask, list of problems)."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        return pd.DataFrame(columns=features), np.zeros(len(df), dtype=bool), [f"missing required columns: {', '.join(missing)}"]
    X = df[features].apply(pd.to_numeric, errors="coerce").astype(float)
    bad = X.isna().any(axis=1).to_numpy().copy()        # writable copy (pandas copy-on-write)
    problems = [f"{int(bad.sum())} row(s) have missing or non-numeric values"] if bad.any() else []
    for f in features:
        lo, hi = PLAUSIBLE.get(f, (-np.inf, np.inf))
        out = ((X[f] < lo) | (X[f] > hi)).fillna(False).to_numpy()
        if out.any():
            problems.append(f"{int(out.sum())} row(s) have an implausible {f} (outside {lo:g} to {hi:g})")
            bad |= out
    return X.reset_index(drop=True), ~bad, problems


def mahalanobis(Z, mean, inv_cov) -> np.ndarray:
    diff = np.asarray(Z, float) - mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, inv_cov, diff), 0.0))


def _bin_counts(values, inner_edges) -> np.ndarray:
    idx = np.searchsorted(inner_edges, values, side="right")
    return np.bincount(idx, minlength=len(inner_edges) + 1)


def psi(expected, actual, eps: float = 1e-4) -> float:
    """Population stability index between two binned distributions (proportions)."""
    e, a = np.clip(np.asarray(expected, float), eps, None), np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(a_sorted, b) -> float:
    """Two-sample Kolmogorov-Smirnov distance D (Massey, 1951); a_sorted must already be sorted."""
    b = np.sort(np.asarray(b, float))
    if b.size == 0:
        return float("nan")
    grid = np.concatenate([a_sorted, b])
    return float(np.max(np.abs(np.searchsorted(a_sorted, grid, side="right") / a_sorted.size
                               - np.searchsorted(b, grid, side="right") / b.size)))


@dataclass
class Reference:
    features: list
    inner_edges: dict
    expected: dict
    sorted_values: dict
    mean: np.ndarray
    inv_cov: np.ndarray
    threshold: float


def build_reference(train: pd.DataFrame, to_model_space, bins: int = 10, quantile: float = 0.99) -> Reference:
    """Summarise the training distribution once: decile bins per feature and a Mahalanobis envelope."""
    features = list(train.columns)
    inner, expected, sorted_values = {}, {}, {}
    for f in features:
        v = train[f].to_numpy(float)
        inner[f] = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
        expected[f] = _bin_counts(v, inner[f]) / v.size
        sorted_values[f] = np.sort(v)
    Z = to_model_space(train.to_numpy(float))
    mean = Z.mean(axis=0)
    inv_cov = np.linalg.inv(np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1]))
    threshold = float(np.quantile(mahalanobis(Z, mean, inv_cov), quantile))
    return Reference(features, inner, expected, sorted_values, mean, inv_cov, threshold)


def drift_table(ref: Reference, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in ref.features:
        v = X[f].to_numpy(float)
        value = psi(ref.expected[f], _bin_counts(v, ref.inner_edges[f]) / max(v.size, 1))
        rows.append({"feature": f, "PSI": value, "KS D": ks_statistic(ref.sorted_values[f], v),
                     "status": next(label for limit, label in PSI_BANDS if value < limit),
                     "training median": float(np.median(ref.sorted_values[f])),
                     "new median": float(np.median(v)) if v.size else float("nan")})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)
