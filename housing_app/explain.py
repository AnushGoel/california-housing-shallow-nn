"""Exact Shapley values for a single prediction against one reference input.

Each player (a feature or a group of features) is switched from the reference value to the actual value in every
possible coalition, and its contributions are averaged with the Shapley weights (Shapley, 1953). With one reference
point this is the "baseline" Shapley value (Sundararajan & Najmi, 2020). For the shallow network, 2^7 = 128 forward
passes give exact values in under a millisecond, so no sampling approximation is needed (Lundberg & Lee, 2017).
"""
from __future__ import annotations

from math import factorial

import numpy as np

MAX_PLAYERS = 12


def shapley_values(predict, x, baseline, groups=None) -> np.ndarray:
    """phi[i] for each player such that predict(baseline) + phi.sum() == predict(x) (the efficiency property)."""
    x, base = np.asarray(x, float).ravel(), np.asarray(baseline, float).ravel()
    if x.shape != base.shape:
        raise ValueError("x and baseline must have the same length")
    groups = [list(g) for g in (groups or [[j] for j in range(x.size)])]
    m = len(groups)
    if m > MAX_PLAYERS:
        raise ValueError(f"exact Shapley values are limited to {MAX_PLAYERS} players")
    coalitions = np.arange(2 ** m)
    members = ((coalitions[:, None] >> np.arange(m)) & 1).astype(bool)          # (2^m, m)
    X = np.tile(base, (2 ** m, 1))
    for i, cols in enumerate(groups):
        rows = np.flatnonzero(members[:, i])
        X[np.ix_(rows, cols)] = x[cols]
    f = np.asarray(predict(X), dtype=float).ravel()
    size = members.sum(axis=1)
    weight = np.array([factorial(s) * factorial(m - s - 1) / factorial(m) for s in range(m)])
    phi = np.zeros(m)
    for i in range(m):
        without = np.flatnonzero(~members[:, i])
        phi[i] = np.sum(weight[size[without]] * (f[without | (1 << i)] - f[without]))
    return phi
