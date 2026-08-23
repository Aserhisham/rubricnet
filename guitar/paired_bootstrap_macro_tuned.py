"""Paired bootstrap: RubricNet against Random Forests tuned on the macro-averaged errors.

Motivation
----------
RubricNet's Optuna study was multi-objective over (validation balanced accuracy,
validation macro-MSE), while the tuned baselines of `guitar/tuned_baselines.py` were
selected on balanced accuracy / accuracy alone. Since the headline claim in the thesis
resolves precisely on the macro-averaged and error-magnitude measures, that asymmetry
favours RubricNet on exactly the metrics it is said to win. This script removes it by
pairing RubricNet against forests selected *directly* on macro-MAE and macro-MSE
(produced by `guitar/tuned_baselines.py --scoring macro_mae,macro_mse`).

Both models are scored on identical resamples of the same 640 test pieces, so the
piece-to-piece variance they share cancels.

Usage
-----
    python -m guitar.tuned_baselines --v5-pruned-collinear2 --n-iter 60 \
        --models random_forest --scoring macro_mae,macro_mse --skip-trivial \
        --out guitar/tuned_baseline_results_macro_selection.json
    python -m guitar.paired_bootstrap_macro_tuned
"""
import json

import numpy as np
from sklearn.metrics import cohen_kappa_score

RUBRICNET_PREDS = "guitar/metric_study_predictions.json"
BASELINE_PREDS = "guitar/tuned_baseline_results_macro_selection.json"
OUT_PATH = "guitar/paired_bootstrap_vs_macro_tuned_rf.json"
N_RESAMPLES = 4000
SEED = 0


def _macro(y, p, power):
    return float(np.mean([np.mean(np.abs(p[y == c] - y[y == c]) ** power) for c in np.unique(y)]))


METRICS = {
    "macro_mse": (lambda y, p: _macro(y, p, 2), False),
    "macro_mae": (lambda y, p: _macro(y, p, 1), False),
    "qwk": (lambda y, p: cohen_kappa_score(y, p, weights="quadratic"), True),
    "mse": (lambda y, p: float(np.mean((y - p) ** 2)), False),
    "mae": (lambda y, p: float(np.mean(np.abs(y - p))), False),
    "balanced_accuracy": (lambda y, p: float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])), True),
    "accuracy": (lambda y, p: float(np.mean(y == p)), True),
}


def main():
    runs = [r for r in json.load(open(RUBRICNET_PREDS)) if r["model"] == "RubricNet" and r["seed"] == 0]
    runs.sort(key=lambda r: r["fold"])
    y_true = np.concatenate([r["y_true"] for r in runs])
    y_rn = np.concatenate([r["y_pred"] for r in runs])

    baselines = json.load(open(BASELINE_PREDS))
    rng = np.random.default_rng(SEED)
    results = {"n_resamples": N_RESAMPLES, "n_pieces": int(len(y_true)),
               "note": "single RubricNet seed (0) against the deterministic forests"}

    for key in ("random_forest[macro_mae]", "random_forest[macro_mse]"):
        folds = sorted(baselines[key]["predictions"], key=lambda p: p["fold"])
        assert all(np.array_equal(np.array(f["y_true"]), np.array(r["y_true"]))
                   for f, r in zip(folds, runs)), "fold ordering mismatch"
        y_rf = np.concatenate([np.array(f["y_pred"]) for f in folds])

        out = {}
        for name, (fn, higher_is_better) in METRICS.items():
            wins = 0
            for _ in range(N_RESAMPLES):
                idx = rng.integers(0, len(y_true), len(y_true))
                a, b = fn(y_true[idx], y_rn[idx]), fn(y_true[idx], y_rf[idx])
                wins += (a > b) if higher_is_better else (a < b)
            out[name] = {"rubricnet": fn(y_true, y_rn), "rf": fn(y_true, y_rf),
                         "p_rubricnet_better": wins / N_RESAMPLES}
            print(f"{key:28s} {name:18s} RN={out[name]['rubricnet']:.4f} "
                  f"RF={out[name]['rf']:.4f} p={out[name]['p_rubricnet_better']:.3f}")
        results[key] = out

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
