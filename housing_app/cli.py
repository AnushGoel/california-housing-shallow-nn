"""Command line:  python -m housing_app <command>   (run `python -m housing_app -h` for the list)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .artifacts import FILES, ArtifactError, stored_name
from .config import APP_ROOT, load_settings
from .service import FEATURES, PredictionService
from .storage import LocalStorage, build_storage


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m housing_app",
                                description="Predictions, explanations, batch scoring and integrity checks for the "
                                            "California housing shallow network.")
    p.add_argument("--artifacts", type=Path, default=None, help="artifact folder (default: $ARTIFACTS_DIR or ./artifacts)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="summarise the trained models")
    sub.add_parser("validate", help="check checksums, the schema and the NumPy-vs-Keras self-check")
    for name, text in (("predict", "price one block group (missing features take typical values)"),
                       ("explain", "exact Shapley breakdown of one prediction")):
        sp = sub.add_parser(name, help=text)
        for f in FEATURES:
            sp.add_argument(f"--{f}", type=float, default=None)
        sp.add_argument("--model", choices=["final", "tuned"], default="final")
        sp.add_argument("--level", type=float, default=0.9, help="coverage of the conformal interval")
        sp.add_argument("--json", action="store_true", help="print JSON instead of text")
    sc = sub.add_parser("score", help="score a CSV of block groups (adds prediction, interval and OOD columns)")
    sc.add_argument("csv", type=Path)
    sc.add_argument("-o", "--output", type=Path, help="write the scored CSV here (default: standard output)")
    sc.add_argument("--model", choices=["final", "tuned"], default="final")
    sc.add_argument("--level", type=float, default=0.9)
    dr = sub.add_parser("drift", help="compare a CSV with the training distribution (PSI and KS)")
    dr.add_argument("csv", type=Path)
    rp = sub.add_parser("report", help="regenerate docs/RESULTS.md, the figures and the README summary")
    rp.add_argument("--check", action="store_true", help="only check that the report matches the artifacts")
    rp.add_argument("--no-figures", action="store_true")
    sub.add_parser("publish", help="copy the artifacts to the configured S3-compatible bucket")
    return p


def _artifacts_dir(args) -> Path:
    return args.artifacts or Path(os.environ.get("ARTIFACTS_DIR", APP_ROOT / "artifacts"))


def main(argv=None, out=None) -> int:
    out = out or sys.stdout
    args = _parser().parse_args(argv)
    folder = _artifacts_dir(args)
    if args.command == "report":
        from . import report
        return report.main(folder, check=args.check, figures=not args.no_figures, out=out)
    try:
        service = PredictionService.from_dir(folder)
    except ArtifactError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    b = service.bundle
    if args.command == "info":
        fin = b.final
        print(f"{'artifacts':<16}{b.source} (fingerprint {b.fingerprint}, checksums {'verified' if b.verified else 'not listed'})", file=out)
        for label, net in b.models.items():
            print(f"{label:<16}{net.describe()}, {net.n_params:,} parameters, weights from epoch {net.best_epoch}", file=out)
        print(f"{'test set':<16}MAE ${fin['test_mae'] * 1e5:,.0f}  RMSE ${fin['test_rmse'] * 1e5:,.0f}  R2 {fin['test_r2']:.3f}", file=out)
        return 0
    if args.command == "validate":
        worst = max(net.self_check() for net in b.models.values())
        ok = worst < 1e-4
        print(f"schema ok, checksums {'verified' if b.verified else 'not listed in a manifest'}, "
              f"NumPy vs Keras max difference {worst:.1e} -> {'PASS' if ok else 'FAIL'}", file=out)
        return 0 if ok else 1
    if args.command in ("predict", "explain"):
        record = {f: getattr(args, f) for f in FEATURES if getattr(args, f) is not None}
        result = service.predict_one(record, args.model, args.level) if args.command == "predict" else service.explain(record, args.model)
        if args.json:
            print(json.dumps(result, indent=2), file=out)
        elif args.command == "predict":
            interval = f"  ({args.level:.0%} interval ${result['lower']:,.0f} to ${result['upper']:,.0f})" if "lower" in result else ""
            flag = "  [outside the training distribution]" if result["out_of_distribution"] else ""
            print(f"predicted median house value ${result['prediction']:,.0f}{interval}{flag}", file=out)
        else:
            print(f"typical block group ${result['baseline']:,.0f}", file=out)
            for c in sorted(result["contributions"], key=lambda c: -abs(c["contribution"])):
                print(f"  {c['feature']:<12}{c['contribution']:+12,.0f}", file=out)
            print(f"prediction          ${result['prediction']:,.0f}", file=out)
        return 0
    if args.command == "score":
        scored, problems = service.predict_frame(pd.read_csv(args.csv), args.model, args.level)
        for p in problems:
            print(f"warning: {p}", file=sys.stderr)
        if args.output:
            scored.to_csv(args.output, index=False)
            print(f"scored {int(scored['valid'].sum())} of {len(scored)} rows -> {args.output}", file=out)
        else:
            scored.to_csv(out, index=False)
        return 0
    if args.command == "drift":
        table, problems = service.drift(pd.read_csv(args.csv))
        for p in problems:
            print(f"warning: {p}", file=sys.stderr)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"), file=out)
        return 0
    if args.command == "publish":
        settings = load_settings(_read_secrets())
        if not settings.bucket:
            print("error: no bucket configured (storage.bucket in .streamlit/secrets.toml or HOUSING_BUCKET)", file=sys.stderr)
            return 2
        source, target = LocalStorage(folder), build_storage(replace(settings, artifacts_backend="s3"), "artifacts")
        for name in [stored_name(source, n) for n in FILES] + ["manifest.json"]:
            if name and source.exists(name):
                target.write_bytes(name, source.read_bytes(name))
                print(f"uploaded {name:<26} -> {target.label}/{name}", file=out)
        return 0
    return 1


def _read_secrets() -> dict:
    path = APP_ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:                  # Python 3.10
        import tomli as tomllib
    return tomllib.loads(path.read_text())
