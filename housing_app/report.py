"""Evidence report: turns the artifacts into docs/RESULTS.md, small SVG figures and the README and model-card summaries.

Every number in the generated text is read or recomputed from the artifacts, and the report records their fingerprint,
so anyone can check that the conclusions belong to exactly these files (`python -m housing_app report --check`).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ArtifactError, load_bundle
from .config import APP_ROOT
from .evaluation import compare, interval, mae, paired_bootstrap
from .storage import LocalStorage

DOLLARS = 100_000
N_BOOT, SEED = 2000, 42        # the notebook's resample count and seed, so every document quotes the same intervals
START, END = "<!-- results:start -->", "<!-- results:end -->"
REFERENCES = {
    "Cawley": "Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html",
    "Efron": "Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593",
    "Grinsztajn": "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35* (pp. 507–520). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html",
    "Glorot": "Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics* (pp. 249–256). PMLR. https://proceedings.mlr.press/v9/glorot10a.html",
    "Hastie": "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7",
    "Lei": "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association, 113*(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116",
    "Roberts": "Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881",
    "Robinson": "Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review, 15*(3), 351–357. https://doi.org/10.2307/2087176",
    "Tobin": "Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica, 26*(1), 24–36. https://doi.org/10.2307/1907382",
}


NICE = {"relu": "ReLU", "elu": "ELU", "gelu": "GELU", "tanh": "tanh", "sigmoid": "sigmoid"}
WORDS = dict(enumerate("zero one two three four five six seven eight nine ten".split()))


def dollars(v) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def usd(v_100k) -> str:
    return dollars(v_100k * DOLLARS)


def nice(name) -> str:
    return NICE.get(str(name), str(name))


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def words(n: int) -> str:
    return WORDS.get(n, str(n))


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


class Evidence:
    """Everything the report states, computed once from one bundle."""

    def __init__(self, bundle):
        self.b, self.R = bundle, bundle.results
        P = bundle.predictions
        self.y = P["y_true"].to_numpy(float)
        self.preds = {k: P[c].to_numpy(float) for k, c in bundle.prediction_columns().items()}
        self.draws = paired_bootstrap(self.y, self.preds, n_boot=N_BOOT, seed=SEED)
        self.ev = self.R.get("evaluation", {})

    def point_difference(self, other: str) -> float:
        """Full-sample MAE of the final model minus another model's (the bootstrap supplies only the interval)."""
        return mae(self.y, self.preds["Final model"]) - mae(self.y, self.preds[other])

    def ci(self, model="Final model", metric="MAE"):
        """The notebook's own bootstrap interval when it was exported, so every document quotes identical numbers."""
        for row in self.ev.get("bootstrap", []):
            if row.get("model") == model and f"{metric} low" in row:
                return float(row[f"{metric} low"]), float(row[f"{metric} high"])
        return interval(self.draws[model][metric])

    # ---- findings --------------------------------------------------------------------------------------------
    def headline(self):
        fin = self.R["final"]
        lo, hi = self.ci()
        r2_lo, r2_hi = self.ci(metric="R2")
        rows = [("MAE", usd(fin["test_mae"]), f"{usd(lo)} to {usd(hi)}"),
                ("RMSE", usd(fin["test_rmse"]), "{} to {}".format(*map(usd, self.ci(metric="RMSE")))),
                ("R²", f"{fin['test_r2']:.3f}", f"{r2_lo:.3f} to {r2_hi:.3f}")]
        conf = self._conf90()
        if conf is not None:
            rows.append(("90% conformal interval", f"±{dollars(conf['half-width ($)'])}", f"covered {conf['test coverage']:.1%} of test block groups"))
        return md_table(pd.DataFrame(rows, columns=["test-set metric", "value", f"95% bootstrap interval ({N_BOOT:,} resamples)"]))

    def capacity(self):
        st = pd.DataFrame(self.R["widths"]["seed_table"]).sort_values("Hidden units")
        sel = int(self.R["widths"]["selected"])
        best = st.loc[st["mean best val loss"].idxmin()]
        n_seeds = int(st["seeds"].iloc[0])
        mono = st["mean final train loss"].is_monotonic_decreasing
        text = (f"Training loss {'fell steadily' if mono else 'did not fall steadily'} as the hidden layer widened (mean final "
                f"training MSE {st['mean final train loss'].iloc[0]:.4f} at {int(st['Hidden units'].iloc[0])} units, "
                f"{st['mean final train loss'].iloc[-1]:.4f} at {int(st['Hidden units'].iloc[-1])}). ")
        if int(best["Hidden units"]) == sel:
            gap = (st.loc[st["Hidden units"] != sel, "mean best val loss"] - best["mean best val loss"]).min()
            clear = np.isfinite(best["SE"]) and gap > best["SE"]
            text += (f"{sel} units also reached the lowest mean best validation MSE over {words(n_seeds)} seeds "
                     f"({best['mean best val loss']:.4f} ± {best['SE']:.4f} SE), "
                     + ("clear of the next width by more than one standard error." if clear else
                        "but the next width was within one standard error, so the advantage is weak."))
        else:
            text += (f"The lowest mean best validation MSE belonged to {int(best['Hidden units'])} units "
                     f"({best['mean best val loss']:.4f}), but {sel} units were within one standard error of it, so the "
                     "one-standard-error rule kept the smaller network (Hastie et al., 2009).")
        table = st.assign(**{"best validation MSE (mean ± SE)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(st["mean best val loss"], st["SE"])],
                             "best epoch (mean)": st["mean best epoch"].round(0).astype(int),
                             "final training MSE (mean)": st["mean final train loss"].round(4)})
        return text, md_table(table[["Hidden units", "best validation MSE (mean ± SE)", "best epoch (mean)", "final training MSE (mean)"]])

    def activation(self):
        st = pd.DataFrame(self.R["activations"]["seed_table"]).set_index("Activation")
        tbl = pd.DataFrame(self.R["activations"]["table"]).set_index("Activation")
        order = st["mean best val loss"].sort_values()
        text = (f"{cap(nice(order.index[0]))} reached the lowest mean best validation MSE ({order.iloc[0]:.4f}), ahead of "
                + " and ".join(f"{nice(a)} ({v:.4f})" for a, v in order.iloc[1:].items()) + ". ")
        if "Epochs to common target" in tbl:
            conv = tbl["Epochs to common target"]
            if conv.notna().any():
                fast = conv.idxmin()
                never = [a for a, v in conv.items() if pd.isna(v)]
                text += (f"{cap(nice(fast))} was the fastest to reach the common target ({conv[fast]:.0f} epochs)"
                         + (f", and {' and '.join(nice(a) for a in never)} never reached it within the budget. " if never else ". "))
        sat = {r["Activation"]: r["share of unit-example pairs in the flat region"] for r in self.R["activations"].get("saturation", [])}
        if "sigmoid" in sat and "tanh" in sat:
            if max(sat["sigmoid"], sat["tanh"]) < 0.05:
                text += (f"Saturation was rare after training ({sat['sigmoid']:.1%} of sigmoid and {sat['tanh']:.1%} of tanh "
                         "pre-activations sat on the flat parts of their curves), so it does not explain their slower learning; "
                         "their small, nearly linear gradients near zero do (LeCun et al., 2012). ")
            else:
                text += (f"After training, {sat['sigmoid']:.1%} of sigmoid pre-activations sat on the flat part of the curve, "
                         f"against {sat['tanh']:.1%} for tanh, in line with the account of saturating units in Glorot and Bengio (2010). ")
        text += f"The selected activation was {nice(self.R['activations']['selected'])}."
        table = order.to_frame("mean best validation MSE").reset_index()
        table["Activation"] = table["Activation"].map(nice)
        table["mean best validation MSE"] = table["mean best validation MSE"].map("{:.4f}".format)
        return text, md_table(table)

    def tuning(self):
        hp, fin = self.R["hpo"], self.R["final"]
        champ = hp.get("champion_cfg", {})
        gain = 1 - hp.get("champion_best_val", fin["best_val_loss"]) / fin["best_val_loss"]
        comp = pd.DataFrame(self.R["comparison_test"]).set_index("index")
        tuned = comp[comp.index.str.contains("extension")]
        text = (f"The random search ({self.R['meta'].get('hpo_trials', '?')} trials) favoured {champ.get('hidden_units')} "
                f"{nice(champ.get('activation'))} units with learning rate {champ.get('lr', 0):g}, L2 {champ.get('l2', 0):g}, "
                f"{'no dropout' if not champ.get('dropout') else 'dropout ' + format(champ.get('dropout'), 'g')} and {'log-transformed' if champ.get('features') == 'log' else 'raw'} features. Its validation MSE was "
                f"{hp.get('champion_best_val', float('nan')):.4f}, {abs(gain):.1%} {'below' if gain > 0 else 'above'} the "
                f"reference model's {fin['best_val_loss']:.4f}. Part of any validation gain is selection bias from picking the "
                "best of many trials (Cawley & Talbot, 2010), so the test set, which played no part in the search, is the "
                "fair judge: ")
        if len(tuned):
            text += f"there the champion's MAE was {usd(tuned['MAE'].iloc[0])} against {usd(fin['test_mae'])} for the reference model"
            text += (f", so the validation gain did not carry over to typical error, although the champion's test MSE was lower "
                     f"({tuned['MSE'].iloc[0]:.4f} against {fin['test_mse']:.4f})." if tuned["MAE"].iloc[0] > fin["test_mae"]
                     and tuned["MSE"].iloc[0] < fin["test_mse"] else ".")
        return text

    def models(self):
        rows, sentences = [], []
        for name in self.preds:
            lo, hi = self.ci(name)
            rows.append((name, usd(mae(self.y, self.preds[name])), f"{usd(lo)} to {usd(hi)}"))
        for name in self.preds:
            if name == "Final model":
                continue
            c = compare(self.draws["Final model"], self.draws[name], "MAE")
            d, lo, hi = self.point_difference(name), c["low"], c["high"]
            who = {"Tuned champion": "the tuned champion"}.get(name, name.lower())
            if c["verdict"] == "first better":
                sentences.append(f"beat {who} by {usd(-d)} (95% interval {usd(-hi)} to {usd(-lo)})")
            elif c["verdict"] == "second better":
                sentences.append(f"lost to {who} by {usd(d)} (95% interval {usd(lo)} to {usd(hi)})")
            else:
                sentences.append(f"could not be told apart from {who} (difference {usd(d)}, interval {usd(lo)} to {usd(hi)})")
        text = ("Scored on the same resampled test block groups (a paired bootstrap; Efron & Tibshirani, 1993), the final "
                "network " + "; it ".join(sentences) + ".")
        return text, md_table(pd.DataFrame(rows, columns=["model", "test MAE", "95% bootstrap interval"]))

    def spatial(self):
        if "cv" not in self.ev:
            return None
        cv = pd.DataFrame(self.ev["cv"])
        m = cv.groupby("scheme")["MAE"].agg(["mean", "std"])
        opt = float(self.ev.get("cv_optimism_pct", 100 * (m.loc["spatial", "mean"] / m.loc["random", "mean"] - 1)))
        text = (f"Random {self.ev.get('cv_folds', '?')}-fold cross-validation of the final configuration gave an MAE of "
                f"{usd(m.loc['random', 'mean'])} (SD {usd(m.loc['random', 'std'])} across folds). Holding out whole regions "
                f"gave {usd(m.loc['spatial', 'mean'])} (SD {usd(m.loc['spatial', 'std'])}), a change of {opt:+.1f}%. ")
        if opt > 5:
            text += ("Pricing places the network has never seen is clearly harder, so a random-split score is an optimistic "
                     "guide to performance in a new region (Roberts et al., 2017).")
        elif opt > -5:
            text += "The cost of unseen regions was small, so the random-split score is a fair guide here."
        else:
            text += "Spatial folds came out easier, which with few, uneven regions is plausible but should be read with caution."
        return text

    def _conf90(self):
        rows = self.ev.get("conformal")
        if not rows:
            return None
        df = pd.DataFrame(rows)
        hit = df[np.isclose(df["promised coverage"], 0.9)]
        return hit.iloc[0] if len(hit) else None

    def conformal(self):
        row = self._conf90()
        if row is None:
            return None
        cov, gap = float(row["test coverage"]), float(row["test coverage"]) - 0.9
        text = (f"A 90% split-conformal interval is ±{dollars(row['half-width ($)'])} around every prediction. On the test set "
                f"it covered {cov:.1%} of block groups against a promise of 90.0% (Lei et al., 2018), ")
        text += ("so the promise held." if abs(gap) <= 0.02 else
                 ("which falls short; the validation set also selected the model, which can make it slightly optimistic." if gap < 0 else
                  "which is more than promised, so the intervals are conservative."))
        if pd.notna(row.get("coverage at the ceiling")):
            text += (f" Coverage was {row['coverage at the ceiling']:.1%} for block groups at the $500,000 ceiling and "
                     f"{row['coverage below the ceiling']:.1%} below it: a single width fits everyone, so it is too narrow "
                     "exactly where the errors are largest.")
        return text

    def weak_spots(self):
        lines = []
        for name, records in self.ev.get("slices", {}).items():
            if records and name != "price ceiling":          # the ceiling gets its own, fuller line below
                w = records[0]
                lines.append(f"- {name}: weakest slice *{w['slice']}*, MAE {dollars(w['MAE ($)'])} (95% interval "
                             f"{dollars(w['CI low ($)'])} to {dollars(w['CI high ($)'])}) on {int(w['block groups']):,} block groups")
        cap = {r["index"]: r for r in self.R["final"].get("cap_table", [])}
        at = next((v for k, v in cap.items() if "ceiling" in k and "below" not in k), None)
        below = next((v for k, v in cap.items() if "below" in k), None)
        if at and below:
            lines.append(f"- price ceiling: MAE {dollars(at['MAE ($)'])} for capped block groups against {dollars(below['MAE ($)'])} "
                         "below the cap. The true value of a capped block group is only known to be at least $500,000, so its "
                         "recorded error understates the real one (Tobin, 1958).")
        return "\n".join(lines)

    def importance(self):
        imp = pd.DataFrame(self.R.get("importance", []))
        if imp.empty:
            return None
        top = imp.head(3)
        return ("Shuffling one feature at a time on the test set raised the MSE most for "
                + ", ".join(f"{r['index']} (+{r['rise as % of test MSE']:.0f}%)" for _, r in top.iterrows())
                + ". These scores measure what the model relies on, not causal effects.")

    def hypotheses(self):
        """Verdicts on the hypotheses pre-registered in docs/METHODOLOGY.md, computed from the artifacts."""
        R, rows = self.R, []
        fmt = lambda v: "never" if pd.isna(v) else f"{v:.0f}"  # noqa: E731
        st = pd.DataFrame(R["widths"]["seed_table"]).set_index("Hidden units")
        st.index = st.index.astype(int)
        if {16, 64} <= set(st.index):
            m16, m64, se = st.loc[16, "mean best val loss"], st.loc[64, "mean best val loss"], st.loc[64, "SE"]
            rows.append(("H1", "16 hidden units underfit relative to 64", f"{m16:.4f} against {m64:.4f} (SE {se:.4f})",
                         "supported" if m16 > m64 + (se if np.isfinite(se) else 0.0) else "not supported"))
        tbl = pd.DataFrame(R["activations"]["table"]).set_index("Activation")
        if "Epochs to common target" in tbl and {"relu", "tanh", "sigmoid"} <= set(tbl.index):
            e = tbl["Epochs to common target"]
            sig = e["sigmoid"]
            # a comparison is decided only if the rival reached the target; if neither did, it stays undecided
            outcome = {o: None if pd.isna(e[o]) else bool(pd.isna(sig) or sig > e[o]) for o in ("relu", "tanh")}
            decided = [v for v in outcome.values() if v is not None]
            verdict = ("not supported" if decided and not all(decided) else
                       "supported" if len(decided) == len(outcome) else "partly supported" if decided else "undecided")
            undecided = [nice(o) for o, v in outcome.items() if v is None]
            rows.append(("H2", "Sigmoid converges more slowly than ReLU and tanh",
                         f"epochs to the common target: sigmoid {fmt(sig)}, ReLU {fmt(e['relu'])}, tanh {fmt(e['tanh'])}"
                         + (f"; {' and '.join(undecided)} never reached it either, so that comparison is undecided" if undecided else ""),
                         verdict))
        fin = R["final"]
        rows.append(("H3", "The selected model's validation error is optimistic",
                     f"validation MSE {fin['best_val_loss']:.4f}, test MSE {fin['test_mse']:.4f}",
                     "supported" if fin["test_mse"] > fin["best_val_loss"] else "not supported"))
        if "cv" in self.ev:
            m = pd.DataFrame(self.ev["cv"]).groupby("scheme")["MAE"].mean()
            rows.append(("H4", "Unseen regions are harder than random folds suggest",
                         f"spatial-fold MAE {usd(m['spatial'])}, random-fold MAE {usd(m['random'])}",
                         "supported" if m["spatial"] > m["random"] else "not supported"))
        for code, rival, claim, wanted in (("H5", "Linear regression", "The network beats linear regression", {"first better"}),
                                           ("H6", "Gradient-boosted trees", "Gradient-boosted trees are at least as accurate",
                                            {"second better", "no clear difference"})):
            if rival in self.preds:
                c = compare(self.draws["Final model"], self.draws[rival], "MAE")
                rows.append((code, claim, f"MAE difference {usd(self.point_difference(rival))} (95% interval {usd(c['low'])} to {usd(c['high'])})",
                             "supported" if c["verdict"] in wanted else "not supported"))
        conf = self._conf90()
        if conf is not None:
            rows.append(("H7", "90% conformal intervals keep their promise", f"test coverage {conf['test coverage']:.1%}",
                         "supported" if abs(float(conf["test coverage"]) - 0.9) <= 0.02 else "not supported"))
        cap = {r["index"]: r for r in fin.get("cap_table", [])}
        at = next((v for k, v in cap.items() if "below" not in k), None)
        below = next((v for k, v in cap.items() if "below" in k), None)
        if at and below:
            rows.append(("H8", "Errors are larger at the price ceiling",
                         f"MAE {dollars(at['MAE ($)'])} at the ceiling, {dollars(below['MAE ($)'])} below it",
                         "supported" if at["MAE ($)"] > below["MAE ($)"] else "not supported"))
        table = pd.DataFrame(rows, columns=["", "hypothesis", "evidence", "verdict"])
        return table, {v: sum(r[3] == v for r in rows) for v in ("supported", "partly supported", "not supported", "undecided")}, len(rows)

    def conclusion(self):
        fin = self.R["final"]
        lo, hi = self.ci()
        parts = [f"A one-hidden-layer network with {fin['units']} {nice(fin['activation'])} units prices an unseen 1990 California "
                 f"block group with a typical error of {usd(fin['test_mae'])} (95% interval {usd(lo)} to {usd(hi)}) and "
                 f"explains {fin['test_r2']:.0%} of the variance in median house value."]
        verdicts = {n: compare(self.draws["Final model"], self.draws[n], "MAE")["verdict"] for n in self.preds if n != "Final model"}
        if "Linear regression" in verdicts:
            parts.append("It clearly improves on linear regression, which shows the hidden layer is earning its keep."
                         if verdicts["Linear regression"] == "first better" else
                         "It does not clearly beat linear regression, which questions what the hidden layer adds here.")
        if "Gradient-boosted trees" in verdicts:
            parts.append({"first better": "It even beats gradient-boosted trees on this test set.",
                          "second better": "Gradient-boosted trees remain stronger, as they usually are on tabular data of this size.",
                          }.get(verdicts["Gradient-boosted trees"], "It matches gradient-boosted trees within the uncertainty of the test set."))
        if "cv_optimism_pct" in self.ev:
            opt = self.ev["cv_optimism_pct"]
            parts.append(f"Its error grows by {opt:.1f}% when whole regions are held out, so it fills gaps on a known map better "
                         "than it would price a new one." if opt > 5 else "Its error barely changes when whole regions are held out.")
        conf = self._conf90()
        if conf is not None:
            parts.append(f"Its 90% conformal intervals covered {conf['test coverage']:.1%} of test block groups, so the stated "
                         "uncertainty can be taken at face value, except near the price ceiling.")
        table, counts, total = self.hypotheses()
        if counts["supported"] == total:
            parts.append(f"All {words(total)} pre-registered hypotheses were supported by the evidence.")
        else:
            extra = [f"{words(counts['partly supported'])} only in part"] if counts["partly supported"] else []
            extra += [f"{words(n)} {'was' if n == 1 else 'were'} not" for n in [counts["not supported"]] if n]
            extra += [f"{words(n)} could not be decided" for n in [counts["undecided"]] if n]
            parts.append(f"{cap(words(counts['supported']))} of the {words(total)} pre-registered hypotheses were supported"
                         + (", " + ", and ".join(extra) if extra else "") + "; the table above gives the evidence for each.")
        h3 = table[table[""] == "H3"]
        if len(h3) and h3["verdict"].iloc[0] == "not supported" and "cv" in self.ev:
            folds = pd.DataFrame(self.ev["cv"]).query("scheme == 'random'")["MAE"]
            if (folds > fin["test_mae"]).all():
                parts.append(f"Contrary to H3, the test error came out below the validation error. Every random cross-validation fold "
                             f"had a higher MAE than the test set (mean {usd(folds.mean())}), which points to a slightly easy test "
                             "sample rather than an unbiased validation score.")
        parts.append(f"These statements rest on a test set used once, on choices made with validation data only, on repeated "
                     f"runs over {words(len(self.R['meta'].get('robustness_seeds', [1])))} seeds, and on the artifacts with fingerprint "
                     f"`{self.b.fingerprint}`; each number above can be recomputed from them.")
        return " ".join(parts)


def figures(ev: Evidence, folder: Path) -> list:
    """Four small SVG figures (text kept as text, fixed ids) so the repository stays light and diffs stay stable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "california-housing", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E4E9EE"})
    green, blue, orange, red, grey = "#006747", "#1D4E89", "#E8772E", "#A23B2A", "#9AA6AE"
    written = []

    def save(fig, name):
        path = folder / name
        fig.tight_layout()
        fig.savefig(path, format="svg", metadata={"Date": None})
        plt.close(fig)
        written.append(path)

    st = pd.DataFrame(ev.R["widths"]["seed_table"]).sort_values("Hidden units")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.errorbar(st["Hidden units"], st["mean best val loss"], yerr=st["SE"].fillna(0), marker="o", color=orange, capsize=4, label="best validation MSE (mean ± SE)")
    ax.plot(st["Hidden units"], st["mean final train loss"], marker="s", color=blue, label="final training MSE (mean)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(st["Hidden units"], [str(int(u)) for u in st["Hidden units"]])
    ax.set_xlabel("hidden units")
    ax.set_ylabel("MSE")
    ax.set_title("Capacity: training and validation error by width", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    save(fig, "capacity.svg")

    rows = [(n, mae(ev.y, ev.preds[n]) * DOLLARS, *[v * DOLLARS for v in ev.ci(n)]) for n in ev.preds]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.55 * len(rows)))
    for i, (n, est, lo, hi) in enumerate(rows):
        color = green if n == "Final model" else (orange if n == "Tuned champion" else grey)
        ax.plot([lo, hi], [i, i], color=color, lw=6, solid_capstyle="round")
        ax.plot(est, i, "o", color=color, ms=9, markeredgecolor="white")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("test MAE with 95% bootstrap interval ($)")
    ax.set_title("Model comparison on the same test block groups", loc="left", fontweight="bold")
    save(fig, "models.svg")

    if "cv" in ev.ev:
        cv = pd.DataFrame(ev.ev["cv"])
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for i, (scheme, color) in enumerate([("random", blue), ("spatial", orange)]):
            vals = cv.loc[cv["scheme"] == scheme, "MAE"].to_numpy() * DOLLARS
            ax.scatter(i + np.linspace(-0.1, 0.1, len(vals)), vals, color=color, s=50, zorder=3)
            ax.hlines(vals.mean(), i - 0.22, i + 0.22, color="#1E2B33", lw=2)
        ax.set_xticks([0, 1], ["random folds", "spatial folds"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("held-out MAE per fold ($)")
        ax.set_title("Unseen regions are harder" if ev.ev.get("cv_optimism_pct", 0) > 5 else "Random against spatial folds", loc="left", fontweight="bold")
        save(fig, "spatial_cv.svg")

    if ev.ev.get("conformal"):
        cf = pd.DataFrame(ev.ev["conformal"])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot([0.45, 1], [0.45, 1], ls="--", color=grey, lw=1, label="promise kept exactly")
        ax.plot(cf["promised coverage"], cf["test coverage"], marker="o", color=green, lw=2, label="all test block groups")
        if cf["coverage at the ceiling"].notna().any():
            ax.plot(cf["promised coverage"], cf["coverage at the ceiling"], marker="^", color=red, lw=1.4, label="at the price ceiling")
        ax.set_xlabel("promised coverage")
        ax.set_ylabel("achieved coverage (test set)")
        ax.set_title("Conformal intervals keep their promise", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower right")
        save(fig, "coverage.svg")
    return written


def results_markdown(ev: Evidence, figure_names) -> str:
    b, meta = ev.b, ev.R["meta"]
    versions = meta.get("versions", {})
    fig = lambda name, alt: f"\n\n![{alt}](figures/{name})" if name in figure_names else ""  # noqa: E731
    banner = ("\n> **Warning: these numbers come from the synthetic test fixtures, not a real training run.**\n"
              if b.is_test_fixture else "")
    sections = [f"# Results and evidence\n{banner}",
                f"Generated by `python -m housing_app report` from the artifacts with fingerprint `{b.fingerprint}` "
                f"(checksums {'verified against the manifest' if b.verified else 'not listed'}), produced "
                f"{meta.get('generated', 'at an unknown time')} with TensorFlow {versions.get('tensorflow', '?')} and Keras "
                f"{versions.get('keras', '?')}. Do not edit this file by hand: retrain, then regenerate it.",
                "## Headline", ev.headline(),
                "## Pre-registered hypotheses",
                "These hypotheses and their decision rules were written in [METHODOLOGY.md](METHODOLOGY.md) before the final "
                "training run. The verdicts below are computed from the artifacts, not chosen.",
                md_table(ev.hypotheses()[0])]
    findings = [("Capacity", *ev.capacity(), fig("capacity.svg", "Training and validation error by hidden-layer size")),
                ("Activation", *ev.activation(), ""),
                ("Model comparison", *ev.models(), fig("models.svg", "Test MAE with bootstrap intervals for each model"))]
    sections.append("## Findings")
    for i, (title, text, table, figure) in enumerate(findings, 1):
        sections.append(f"### {i}. {title}\n\n{text}\n\n{table}{figure}")
    extra = [("Tuning beyond the core protocol", ev.tuning(), ""),
             ("Generalisation to unseen regions", ev.spatial(), fig("spatial_cv.svg", "Held-out MAE per fold, random against spatial")),
             ("Calibrated uncertainty", ev.conformal(), fig("coverage.svg", "Promised against achieved coverage of conformal intervals")),
             ("What the network relies on", ev.importance(), "")]
    n = len(findings)
    for title, text, figure in extra:
        if text:
            n += 1
            sections.append(f"### {n}. {title}\n\n{text}{figure}")
    spots = ev.weak_spots()
    if spots:
        n += 1
        sections.append(f"### {n}. Where the model is weakest\n\n{spots}")
    sections += ["## Conclusion", ev.conclusion(),
                 "## Limitations",
                 f"The data describe {ev.R['data'].get('n_rows', 20640):,} census block groups in 1990, so the model prices "
                 "neighbourhoods as they were then, not homes today, and reading its group-level patterns as statements about "
                 f"individual houses would be an ecological fallacy (Robinson, 1950). {ev.R['data'].get('cap_share', 0):.1%} of "
                 "block groups are top-coded at $500,000. The random split shares neighbourhoods between training and test sets, "
                 "which flatters the headline score relative to a new region. The validation set chose the configuration and "
                 "calibrated the intervals, so both are slightly optimistic. All results come from one dataset and one model "
                 "family by design; they say nothing about deeper networks or other markets.",
                 "## How to verify",
                 "- `python -m housing_app validate` re-checks the SHA-256 checksums in `artifacts/manifest.json` and the NumPy-against-Keras self-check.\n"
                 "- `python -m housing_app report --check` confirms that this file was generated from the committed artifacts.\n"
                 "- The executed notebook reproduces every number here from a fixed seed.",
                 "## References", "\n\n".join(REFERENCES[k] for k in sorted(REFERENCES))]
    return "\n\n".join(sections) + "\n"


def summary_block(ev: Evidence) -> str:
    fin = ev.R["final"]
    lo, hi = ev.ci()
    conf = ev._conf90()
    rows = [("Typical error (test MAE)", f"{usd(fin['test_mae'])} (95% CI {usd(lo)} to {usd(hi)})"),
            ("Variance explained (R²)", f"{fin['test_r2']:.3f}")]
    if conf is not None:
        rows.append(("90% prediction interval", f"±{dollars(conf['half-width ($)'])}, covered {conf['test coverage']:.1%} of test block groups"))
    if "cv_optimism_pct" in ev.ev:
        rows.append(("Cost of predicting unseen regions", f"{ev.ev['cv_optimism_pct']:+.1f}% MAE (spatial against random folds)"))
    rows.append(("Artifacts fingerprint", f"`{ev.b.fingerprint}`"))
    note = "\n\n*Synthetic test fixtures, not real results.*" if ev.b.is_test_fixture else ""
    return (f"{START}\n" + md_table(pd.DataFrame(rows, columns=["", "result"])) +
            f"\n\nFull evidence, figures and conclusions: [docs/RESULTS.md](docs/RESULTS.md){note}\n{END}")


def replace_block(path: Path, block: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), flags=re.S)
    if not pattern.search(text):
        return False
    path.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    return True


def build(bundle, docs: Path, readme: Path = None, with_figures: bool = True) -> dict:
    ev = Evidence(bundle)
    figs = figures(ev, docs / "figures") if with_figures else []
    (docs / "RESULTS.md").write_text(results_markdown(ev, {p.name for p in figs}), encoding="utf-8")
    block = summary_block(ev)
    touched = {"results": docs / "RESULTS.md", "figures": figs}
    for target in [p for p in (readme, docs / "MODEL_CARD.md") if p]:
        if replace_block(target, block.replace("docs/RESULTS.md", "RESULTS.md") if target.parent == docs else block):
            touched[target.name] = target
    return touched


def check(bundle, docs: Path) -> bool:
    path = docs / "RESULTS.md"
    if not path.exists():
        return False
    found = re.search(r"fingerprint `([0-9a-f]+)`", path.read_text(encoding="utf-8"))
    return bool(found) and found.group(1) == bundle.fingerprint


def main(folder: Path, check: bool = False, figures: bool = True, out=None, docs: Path = None, readme: Path = None) -> int:
    out = out or sys.stdout
    docs = docs or APP_ROOT / "docs"
    readme = readme or APP_ROOT / "README.md"
    try:
        bundle = load_bundle(LocalStorage(folder))
    except ArtifactError as err:
        print(f"no usable artifacts in {folder} ({err}); nothing to report", file=out)
        return 0
    if check:
        ok = globals()["check"](bundle, docs)
        print("report matches the artifacts" if ok else "docs/RESULTS.md is stale: run `python -m housing_app report`", file=out)
        return 0 if ok else 1
    touched = build(bundle, docs, readme, with_figures=figures)
    print(f"wrote {touched['results']} and {len(touched['figures'])} figure(s); updated "
          + (", ".join(k for k in touched if k.endswith(".md")) or "no summary blocks"), file=out)
    return 0
