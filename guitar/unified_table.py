"""One comparison table for every model in the thesis, scored by one implementation.

Why this exists
---------------
The thesis currently compares models across tables that differ in more than the model:
the generation table uses the 716-piece dataset for V1--V3 and the 640-piece one from
V5 on; the baselines are hardcoded in one table and searched in another; within the
searched table the Random Forest row is selected on `accuracy` while the decision
tree, extra trees and gradient boosting rows are selected on `balanced_accuracy`; the
macro-averaged and chance-corrected measures are reported only for the final
configuration; and RubricNet is averaged over 3 seeds x 5 folds while every baseline
is a single deterministic pass. Several of those differences move the numbers by more
than the effects being claimed.

This script removes all of them at once. Every model dumps its raw per-fold test
predictions; this module reads those dumps and computes the identical metric suite for
every row, so the only thing that varies between rows is the model.

Inputs are prediction dumps written by:
  * guitar/train_guitar_rubricnet.py   -> results["predictions"]
  * guitar/tuned_baselines.py          -> results[model]["predictions"]
  * guitar/run_fuzzy_baselines.py      -> results[method]["per_fold"][i]["y_true"/"y_pred"]

Usage
-----
    python -m guitar.unified_table --out guitar/unified_comparison.json
    python -m guitar.unified_table --latex
"""

import argparse
import json
import os
import sys
from statistics import mean, stdev

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
)

sys.path.insert(0, ".")

from guitar.prepare_splits import NUM_CLASSES

LABELS = list(range(NUM_CLASSES))

# Order matters: this is the column order of the emitted table.
METRICS = [
    ("accuracy", "Acc", "max"),
    ("balanced_accuracy", "BalAcc", "max"),
    ("acc_plus_minus_1", "Acc+-1", "max"),
    ("mae", "MAE", "min"),
    ("mse", "MSE", "min"),
    ("macro_mae", "MacMAE", "min"),
    ("macro_mse", "MacMSE", "min"),
    ("kendall_tau_b", "tau-b", "max"),
    ("spearman_rho", "rho", "max"),
    ("cohen_kappa", "kappa", "max"),
    ("linear_weighted_kappa", "LWK", "max"),
    ("quadratic_weighted_kappa", "QWK", "max"),
]


def _macro_error(y_true, y_pred, power):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(
        np.mean([
            np.mean(np.abs(y_pred[y_true == c] - y_true[y_true == c]) ** power)
            for c in np.unique(y_true)
        ])
    )


def score_all(y_true, y_pred):
    """The single metric implementation every row in the table is scored by."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tau = kendalltau(y_true, y_pred).statistic
    rho = spearmanr(y_true, y_pred).statistic

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "acc_plus_minus_1": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "macro_mae": _macro_error(y_true, y_pred, 1),
        "macro_mse": _macro_error(y_true, y_pred, 2),
        "kendall_tau_b": 0.0 if np.isnan(tau) else float(tau),
        "spearman_rho": 0.0 if np.isnan(rho) else float(rho),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
        "linear_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="linear")
        ),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")
        ),
    }


# --------------------------------------------------------------------------------
# Loaders: each returns a list of runs, one per (seed, fold)
# --------------------------------------------------------------------------------
def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_rubricnet(path, label="RubricNet (25 desc.)"):
    d = _load_json(path)
    if not d or "predictions" not in d:
        return {}
    return {label: d["predictions"]}


def load_tuned_baselines(path, only=None, rename=None):
    d = _load_json(path)
    if not d:
        return {}
    rename = rename or {}
    out = {}
    for key, val in d.items():
        if not isinstance(val, dict) or "predictions" not in val:
            continue
        if only and key not in only:
            continue
        out[rename.get(key, key)] = val["predictions"]
    return out


def load_fuzzy(path, rename=None):
    d = _load_json(path)
    if not d:
        return {}
    rename = rename or {
        "complete_search": "Fuzzy rules (complete search)",
        "fuzzy_pattern_tree": "Fuzzy pattern tree",
    }
    out = {}
    for key in ("complete_search", "fuzzy_pattern_tree"):
        entry = d.get(key)
        if not entry:
            continue
        folds = [f for f in entry.get("per_fold", []) if "y_pred" in f]
        if not folds:
            continue
        out[rename.get(key, key)] = [
            {"fold": i, "seed": 0, "y_true": f["y_true"], "y_pred": f["y_pred"]}
            for i, f in enumerate(folds)
        ]
    return out


# --------------------------------------------------------------------------------
def aggregate(runs):
    """Mean +- std over (seed, fold) runs, plus the pooled-over-everything value."""
    per_run = [score_all(r["y_true"], r["y_pred"]) for r in runs]
    agg = {}
    for key, _, _ in METRICS:
        vals = [m[key] for m in per_run]
        agg[key] = {
            "mean": float(mean(vals)),
            "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
        }

    pooled_true = [v for r in runs for v in r["y_true"]]
    pooled_pred = [v for r in runs for v in r["y_pred"]]
    agg["_pooled"] = score_all(pooled_true, pooled_pred)
    agg["_n_runs"] = len(runs)
    agg["_n_pieces_per_run"] = len(runs[0]["y_true"]) if runs else 0
    agg["_seeds"] = sorted({int(r.get("seed", 0)) for r in runs})
    return agg


def print_table(table, pooled=False):
    name_w = max(len(n) for n in table) + 2
    head = f"{'model':<{name_w}}{'runs':>6}" + "".join(f"{lab:>10}" for _, lab, _ in METRICS)
    print(head)
    print("-" * len(head))

    best = {}
    for key, _, direction in METRICS:
        vals = {n: (t["_pooled"][key] if pooled else t[key]["mean"]) for n, t in table.items()}
        best[key] = (max if direction == "max" else min)(vals, key=vals.get)

    for name, t in table.items():
        row = f"{name:<{name_w}}{t['_n_runs']:>6}"
        for key, _, _ in METRICS:
            v = t["_pooled"][key] if pooled else t[key]["mean"]
            row += f"{v:>9.4f}" + ("*" if best[key] == name else " ")
        print(row)
    print("\n('*' marks the best value in the column; "
          f"{'pooled over all runs' if pooled else 'mean over runs'})")


def emit_latex(table):
    print("\n% --- unified comparison table ---")
    cols = "l" + "c" * len(METRICS)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    print("\\textbf{Model} & " + " & ".join(f"\\textbf{{{lab}}}" for _, lab, _ in METRICS) + " \\\\")
    print("\\midrule")
    best = {}
    for key, _, direction in METRICS:
        vals = {n: t[key]["mean"] for n, t in table.items()}
        best[key] = (max if direction == "max" else min)(vals, key=vals.get)
    for name, t in table.items():
        cells = []
        for key, _, _ in METRICS:
            m, s = t[key]["mean"], t[key]["std"]
            cell = f"{m:.3f} $\\pm$ {s:.3f}"
            cells.append(f"\\textbf{{{cell}}}" if best[key] == name else cell)
        print(f"{name} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rubricnet", default="guitar/rubricnet_results_v5_pruned_collinear2.json")
    ap.add_argument("--baselines", default="guitar/unified_baselines.json")
    ap.add_argument("--fuzzy", default="guitar/fuzzy_results_v5_pruned_collinear2.json")
    ap.add_argument("--out", default="guitar/unified_comparison.json")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--pooled", action="store_true",
                    help="print values pooled over all runs instead of mean over runs")
    args = ap.parse_args()

    sources = {}
    sources.update(load_tuned_baselines(args.baselines))
    sources.update(load_fuzzy(args.fuzzy))
    sources.update(load_rubricnet(args.rubricnet))

    if not sources:
        raise SystemExit("no prediction dumps found -- run the model scripts first")

    table = {}
    sizes = {}
    for name, runs in sources.items():
        table[name] = aggregate(runs)
        sizes[name] = table[name]["_n_pieces_per_run"] * 0 + sum(len(r["y_true"]) for r in runs) // max(
            len({r["fold"] for r in runs}), 1
        )

    print(f"Models: {len(table)}   metric suite: {len(METRICS)} measures, one implementation\n")
    print_table(table, pooled=args.pooled)

    n_test = {n: sum(len(r["y_true"]) for r in runs) // max(len(t["_seeds"]), 1)
              for (n, runs), t in zip(sources.items(), table.values())}
    odd = {n: v for n, v in n_test.items() if v != max(n_test.values())}
    if odd:
        print(f"\nWARNING: these rows do not cover the same number of test pieces: {odd}")
    else:
        print(f"\nAll rows scored on the same {max(n_test.values())} test pieces per seed.")

    if args.latex:
        emit_latex(table)

    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
