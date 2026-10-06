"""Split-conformal prediction intervals: distribution-free, finite-sample marginal coverage.

With n exchangeable calibration scores s_i = |y_i - f(x_i)|, the interval f(x) +/- q, where q is the
ceil((n + 1) * level)-th smallest score, covers a new y with probability at least `level` (Lei et al., 2018).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(scores, level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between 0 and 1")
    s = np.sort(np.asarray(scores, dtype=float))
    if s.size == 0:
        raise ValueError("no calibration scores")
    k = int(np.ceil((s.size + 1) * level))
    return float(s[k - 1]) if k <= s.size else float("inf")


def coverage(y, pred, scores, level: float) -> float:
    q = conformal_quantile(scores, level)
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float)) <= q))


def coverage_table(y, pred, scores, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95), groups=None) -> pd.DataFrame:
    """Promised against achieved coverage, overall and (optionally) for each group label."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rows = []
    for level in levels:
        q = conformal_quantile(scores, level)
        hit = np.abs(y - pred) <= q
        row = {"promised": level, "half_width": q, "achieved": float(hit.mean())}
        if groups is not None:
            for g in pd.unique(groups):
                row[str(g)] = float(hit[np.asarray(groups) == g].mean())
        rows.append(row)
    return pd.DataFrame(rows)
