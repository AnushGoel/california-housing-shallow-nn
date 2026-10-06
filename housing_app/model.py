"""Serving-time inference for the exported one-hidden-layer network, in NumPy (no TensorFlow needed)."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np


def erf(x):
    """Abramowitz and Stegun 7.1.26; absolute error below 1.5e-7, enough for GELU at inference."""
    x = np.asarray(x, dtype=np.float64)
    sign, a = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-a * a))


ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "elu": lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0.0))),
    "gelu": lambda z: 0.5 * z * (1.0 + erf(z / np.sqrt(2.0))),
    "linear": lambda z: z,
}


@dataclass(frozen=True, eq=False)
class ShallowNet:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    activation: str
    feature_mode: str
    log_idx: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    feature_names: tuple
    best_epoch: Optional[int] = None
    check_raw: Optional[np.ndarray] = None
    check_pred: Optional[np.ndarray] = None
    cal_residuals: Optional[np.ndarray] = None   # absolute validation residuals, for conformal intervals

    def __post_init__(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.feature_mode not in ("raw", "log"):
            raise ValueError(f"unknown feature mode {self.feature_mode!r}")
        n_in, n_hidden = len(self.feature_names), self.b1.shape[0]
        if self.W1.shape != (n_in, n_hidden) or self.W2.shape[0] != n_hidden:
            raise ValueError("weight shapes do not match the feature list")

    @classmethod
    def from_npz(cls, source) -> "ShallowNet":
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        with np.load(source, allow_pickle=False) as z:
            arr = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
            return cls(W1=arr("W1"), b1=arr("b1"), W2=arr("W2"), b2=arr("b2"), activation=str(z["activation"]),
                       feature_mode=str(z["feature_mode"]), log_idx=np.asarray(z["log_idx"], dtype=int),
                       scaler_mean=arr("scaler_mean"), scaler_scale=arr("scaler_scale"),
                       feature_names=tuple(str(n) for n in z["feature_names"]),
                       best_epoch=int(z["best_epoch"]) if "best_epoch" in z.files else None,
                       check_raw=arr("check_raw") if "check_raw" in z.files else None,
                       check_pred=arr("check_pred") if "check_pred" in z.files else None,
                       cal_residuals=arr("cal_residuals") if "cal_residuals" in z.files else None)

    @property
    def n_units(self) -> int:
        return int(self.b1.shape[0])

    @property
    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def describe(self) -> str:
        return f"{self.n_units} {self.activation} units, {self.feature_mode} features"

    def inputs(self, X_raw) -> np.ndarray:
        X = np.array(X_raw, dtype=np.float64, ndmin=2, copy=True)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {X.shape[1]}")
        if self.feature_mode == "log":
            X[:, self.log_idx] = np.log1p(X[:, self.log_idx])
        return (X - self.scaler_mean) / self.scaler_scale

    def hidden(self, X_raw):
        Z = self.inputs(X_raw) @ self.W1 + self.b1
        return Z, ACTIVATIONS[self.activation](Z)

    def predict(self, X_raw) -> np.ndarray:
        _, A = self.hidden(X_raw)
        return (A @ self.W2 + self.b2).ravel()

    def contributions(self, x_raw) -> np.ndarray:
        """Each hidden unit's contribution (activation x output weight) to one prediction."""
        _, A = self.hidden(x_raw)
        return A[0] * self.W2[:, 0]

    @property
    def has_intervals(self) -> bool:
        return self.cal_residuals is not None and len(self.cal_residuals) > 0

    def interval(self, X_raw, level: float = 0.9):
        """Split-conformal interval (lower, upper) around each prediction, in target units."""
        from .uncertainty import conformal_quantile
        if not self.has_intervals:
            raise ValueError("this model was exported without calibration residuals")
        pred = self.predict(X_raw)
        q = conformal_quantile(self.cal_residuals, level)
        return pred - q, pred + q

    def self_check(self) -> float:
        """Largest difference from the predictions Keras made in the notebook (nan if none were exported)."""
        if self.check_raw is None or self.check_pred is None:
            return float("nan")
        return float(np.max(np.abs(self.predict(self.check_raw) - self.check_pred)))
