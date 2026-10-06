"""The contract between the training notebook and everything that serves it: files, checksums and validation."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass

import pandas as pd

from .model import ShallowNet

FILES = ("results.json", "histories.json", "dataset.csv", "test_predictions.csv", "model_final.npz", "model_tuned.npz")
COMPRESSIBLE = {"histories.json", "dataset.csv", "test_predictions.csv"}
MODEL_LABELS = {"final": "Final model", "tuned": "Tuned champion"}
PREDICTION_COLUMNS = {"Final model": "y_pred_final", "Tuned champion": "y_pred_tuned",
                      "Linear regression": "y_pred_linear", "Gradient-boosted trees": "y_pred_hgb"}
RESULT_KEYS = {
    "meta": ("seed", "epochs", "batch_size"),
    "data": ("features", "target", "split_sizes"),
    "final": ("units", "activation", "best_epoch", "best_val_loss", "test_mae", "test_rmse", "test_r2"),
    "widths": ("table", "seed_table", "selected"),
    "activations": ("table", "seed_table", "selected"),
    "hpo": ("trials", "champion_trial"),
    "comparison_test": (),
    "importance": (),
}


class ArtifactError(RuntimeError):
    """The artifacts are missing, unreadable, altered, or do not match the expected contract."""


def stored_name(storage, name: str):
    """The file actually stored for a logical artifact: the gzip variant if present, else the plain one."""
    if name in COMPRESSIBLE and storage.exists(name + ".gz"):
        return name + ".gz"
    return name if storage.exists(name) else None


def signature(storage) -> str:
    """Cheap change detector for caches: the manifest's version, or every stored file's version."""
    try:
        if storage.exists("manifest.json"):
            return storage.version("manifest.json")
        return "|".join(storage.version(stored_name(storage, n) or n) for n in FILES)
    except Exception:
        return "unavailable"


@dataclass(eq=False)
class Bundle:
    results: dict
    histories: dict
    dataset: pd.DataFrame
    predictions: pd.DataFrame
    models: dict
    source: str
    fingerprint: str = ""
    verified: bool = False

    @property
    def features(self) -> list:
        return list(self.results["data"]["features"])

    @property
    def target(self) -> str:
        return self.results["data"]["target"]

    @property
    def final(self) -> dict:
        return self.results["final"]

    @property
    def meta(self) -> dict:
        return self.results["meta"]

    @property
    def has_evaluation(self) -> bool:
        return "evaluation" in self.results

    @property
    def is_test_fixture(self) -> bool:
        return "test-double" in str(self.meta.get("versions", {}).get("tensorflow", ""))

    @property
    def final_model(self) -> ShallowNet:
        return self.models[MODEL_LABELS["final"]]

    def prediction_columns(self) -> dict:
        return {label: col for label, col in PREDICTION_COLUMNS.items() if col in self.predictions.columns}


def validate(results: dict, histories: dict, dataset: pd.DataFrame, predictions: pd.DataFrame) -> list:
    problems = []
    for section, keys in RESULT_KEYS.items():
        if section not in results:
            problems.append(f"results.json has no '{section}' section")
            continue
        problems += [f"results.json '{section}' has no '{k}'" for k in keys if k not in results[section]]
    features = list(results.get("data", {}).get("features", []))
    target = results.get("data", {}).get("target")
    problems += [f"dataset.csv has no column '{c}'" for c in features + [target, "split"] if c and c not in dataset.columns]
    problems += [f"test_predictions.csv has no column '{c}'" for c in features + ["y_true", "y_pred_final", "y_pred_tuned"]
                 if c not in predictions.columns]
    if not histories:
        problems.append("histories.json is empty")
    return problems


def load_bundle(storage, verify: bool = True) -> Bundle:
    names = {name: stored_name(storage, name) for name in FILES}
    missing = [name for name, stored in names.items() if stored is None]
    if missing:
        raise ArtifactError(f"missing in {storage.label}: {', '.join(missing)}")
    raw = {name: storage.read_bytes(stored) for name, stored in names.items()}
    digests = {names[n]: hashlib.sha256(b).hexdigest() for n, b in raw.items()}
    verified = False
    if verify and storage.exists("manifest.json"):
        try:
            listed = json.loads(storage.read_bytes("manifest.json"))["files"]
        except (ValueError, KeyError) as err:
            raise ArtifactError(f"manifest.json in {storage.label} is unreadable: {err}") from err
        altered = [f for f, digest in digests.items() if f in listed and listed[f]["sha256"] != digest]
        if altered:
            raise ArtifactError(f"checksum mismatch for {', '.join(altered)}: the files differ from what the notebook wrote")
        verified = all(f in listed for f in digests)
    fingerprint = hashlib.sha256("".join(f"{f}:{d};" for f, d in sorted(digests.items())).encode()).hexdigest()[:12]
    try:
        text = {n: gzip.decompress(b) if names[n].endswith(".gz") else b for n, b in raw.items()}
        results = json.loads(text["results.json"])
        histories = json.loads(text["histories.json"])
        dataset = pd.read_csv(io.BytesIO(text["dataset.csv"]))
        predictions = pd.read_csv(io.BytesIO(text["test_predictions.csv"]))
        models = {MODEL_LABELS[k]: ShallowNet.from_npz(text[f"model_{k}.npz"]) for k in ("final", "tuned")}
    except (ValueError, KeyError, OSError) as err:
        raise ArtifactError(f"could not read the artifacts in {storage.label}: {err}") from err
    problems = validate(results, histories, dataset, predictions)
    if problems:
        raise ArtifactError("; ".join(problems))
    return Bundle(results, histories, dataset, predictions, models, storage.label, fingerprint, verified)
