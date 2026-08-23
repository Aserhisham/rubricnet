"""Per-difficulty-band breakdown of the final model, from the stored predictions.

Chapter 5 reports one accuracy and one MAE over the whole test set. On a distribution
where the three easiest bands hold half the corpus, that is predominantly a statement
about easy repertoire, and the claim that the model's errors concentrate at the hard end
is supported only by a single seed's confusion matrix. This computes the breakdown
directly, pooled over the 3 seeds x 5 folds of predictions already dumped by
train_guitar_rubricnet.py, for RubricNet and for the tuned proportional-odds baseline.

Three readings per band, because no single one is adequate on unequal bands:
  accuracy      fraction of the band's pieces predicted exactly
  MAE           mean |predicted - true| over the band, in bin indices
  RAE           the band's summed absolute error divided by the error a
                global-mean predictor would make on the same pieces (so a value above 1
                means the model does worse on that band than predicting the corpus mean)

RAE is reported because it was asked for, and read with care: a band whose grade sits
near the corpus mean has a small denominator, so its RAE is inflated by the band's
position on the scale rather than by the model. Band 3 (the single grade 8) is the
clearest instance.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUBRICNET_PATH = "guitar/rubricnet_results_v5_pruned_collinear2.json"
BASELINE_PATH = "guitar/proportional_odds_results.json"
OUT_PATH = "guitar/per_level_breakdown.json"

# Frozen equal-frequency bin edges (Section 4.1.4), as raw GuitarBurst grade ranges.
BAND_GRADES = ["1-3", "4-5", "6-7", "8", "9-10", "11-12", "13-15", "16-20"]


def load_predictions(path, key="predictions", subkey=None):
    with open(path) as f:
        data = json.load(f)
    if subkey:
        data = data[subkey]
    dump = data.get(key)
    if not dump:
        return None
    y_true, y_pred = [], []
    for entry in dump:
        y_true.extend(entry["y_true"])
        y_pred.extend(entry["y_pred"])
    return np.asarray(y_true), np.asarray(y_pred)


def breakdown(y_true, y_pred):
    global_mean = y_true.mean()
    rows = []
    for c in range(8):
        mask = y_true == c
        n = int(mask.sum())
        if n == 0:
            continue
        err = np.abs(y_pred[mask] - y_true[mask])
        denom = np.abs(y_true[mask] - global_mean).sum()
        rows.append({
            "band": c,
            "grades": BAND_GRADES[c],
            "n": n,
            "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            "acc_plus_minus_1": float((err <= 1).mean()),
            "mae": float(err.mean()),
            "rae": float(err.sum() / denom) if denom else float("nan"),
            "mean_prediction": float(y_pred[mask].mean()),
        })
    return rows


def main():
    out = {}
    targets = (
        ("rubricnet", RUBRICNET_PATH, None),
        ("proportional_odds", BASELINE_PATH, "proportional_odds[macro_mae]"),
    )
    for name, path, subkey in targets:
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found")
            continue
        loaded = load_predictions(path, subkey=subkey)
        if loaded is None:
            print(f"skip {name}: no prediction dump in {path}")
            continue
        y_true, y_pred = loaded
        rows = breakdown(y_true, y_pred)
        out[name] = {"n_scored": int(len(y_true)), "bands": rows}
        print(f"\n{name}  ({len(y_true)} scored predictions)")
        print(f"{'band':>4} {'grades':>7} {'n':>5} {'acc':>7} {'acc+-1':>7} {'MAE':>7} {'RAE':>7} {'mean pred':>10}")
        for r in rows:
            print(f"{r['band']:>4} {r['grades']:>7} {r['n']:>5} {r['accuracy']:>7.3f} "
                  f"{r['acc_plus_minus_1']:>7.3f} {r['mae']:>7.3f} {r['rae']:>7.3f} {r['mean_prediction']:>10.2f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
