"""One facade over the trained bundle, shared by the dashboard, the command line and the REST API."""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from .artifacts import MODEL_LABELS, Bundle, load_bundle
from .explain import shapley_values
from .monitoring import build_reference, drift_table, mahalanobis, validate_frame
from .storage import LocalStorage
from .uncertainty import conformal_quantile

DOLLARS = 100_000
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
LOCATION = ("Latitude", "Longitude")


class PredictionService:
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self._references = {}

    @classmethod
    def from_dir(cls, path) -> "PredictionService":
        return cls(load_bundle(LocalStorage(path)))

    @property
    def features(self) -> list:
        return self.bundle.features

    def model(self, key: str = "final"):
        if key not in MODEL_LABELS:
            raise ValueError(f"model must be one of {sorted(MODEL_LABELS)}")
        return self.bundle.models[MODEL_LABELS[key]]

    @cached_property
    def typical(self) -> pd.Series:
        """Training medians: the reference block group for explanations and for filling missing inputs."""
        ds = self.bundle.dataset
        return ds[ds["split"] == "train"][self.features].median()

    def reference(self, key: str = "final"):
        if key not in self._references:
            ds = self.bundle.dataset
            self._references[key] = build_reference(ds[ds["split"] == "train"][self.features], self.model(key).inputs)
        return self._references[key]

    def complete(self, record: dict) -> dict:
        unknown = set(record) - set(self.features)
        if unknown:
            raise ValueError(f"unknown feature(s): {', '.join(sorted(unknown))}")
        return {f: float(record[f]) if record.get(f) is not None else float(self.typical[f]) for f in self.features}

    def predict_frame(self, df: pd.DataFrame, key: str = "final", level: float = 0.9):
        """Validated predictions in dollars with conformal intervals and out-of-distribution flags."""
        net, ref = self.model(key), self.reference(key)
        X, ok, problems = validate_frame(df, self.features)
        out = df.reset_index(drop=True).copy()
        out["valid"] = ok
        pred, dist = np.full(len(out), np.nan), np.full(len(out), np.nan)
        if ok.any():
            rows = X[ok].to_numpy(float)
            pred[ok] = net.predict(rows)
            dist[ok] = mahalanobis(net.inputs(rows), ref.mean, ref.inv_cov)
        out["prediction"] = pred * DOLLARS
        if net.has_intervals:
            q = conformal_quantile(net.cal_residuals, level)
            out["lower"], out["upper"] = np.clip(pred - q, 0, None) * DOLLARS, (pred + q) * DOLLARS
        out["ood_distance"] = dist
        out["out_of_distribution"] = np.where(ok, dist > ref.threshold, False)
        return out, problems

    def predict_one(self, record: dict, key: str = "final", level: float = 0.9) -> dict:
        frame, problems = self.predict_frame(pd.DataFrame([self.complete(record)]), key, level)
        if problems:
            raise ValueError("; ".join(problems))
        row = frame.iloc[0]
        result = {"model": key, "prediction": float(row["prediction"]), "out_of_distribution": bool(row["out_of_distribution"]),
                  "ood_distance": float(row["ood_distance"]), "inputs": self.complete(record)}
        if "lower" in frame:
            result.update(level=level, lower=float(row["lower"]), upper=float(row["upper"]))
        return result

    def explain(self, record: dict, key: str = "final") -> dict:
        """Exact Shapley breakdown against the typical block group; Latitude and Longitude act as one player."""
        net, inputs = self.model(key), self.complete(record)
        x = np.array([inputs[f] for f in self.features])
        base = self.typical.to_numpy(float)
        names = [f for f in self.features if f not in LOCATION] + ["location"]
        groups = [[self.features.index(f)] for f in names[:-1]] + [[self.features.index(f) for f in LOCATION]]
        phi = shapley_values(net.predict, x, base, groups)
        return {"model": key, "baseline": float(net.predict(base[None, :])[0]) * DOLLARS,
                "prediction": float(net.predict(x[None, :])[0]) * DOLLARS,
                "contributions": [{"feature": n, "contribution": float(p) * DOLLARS} for n, p in zip(names, phi)]}

    def drift(self, df: pd.DataFrame):
        X, ok, problems = validate_frame(df, self.features)
        return drift_table(self.reference("final"), X[ok]) if ok.any() else pd.DataFrame(), problems
