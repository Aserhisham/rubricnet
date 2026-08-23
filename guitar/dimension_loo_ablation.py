"""Leave-one-dimension-out ablation over the three descriptor dimensions.

Why this exists
---------------
Section 6.3 claims "the right-hand dimension, weak in isolation, still contributes
signal the other dimensions do not carry". Table 5.4 cannot support that: it reports
each dimension *alone* and all three together, and "25 descriptors beat 6" is close to
a statement about feature count. The claim is about complementarity, and the experiment
that tests complementarity is leaving each dimension out of the full set.

This script runs both halves under one protocol: the three single-dimension sets (to
reproduce the published table) and the three leave-one-out sets (new). Random Forest,
matching Section 5.2.3's choice to isolate the feature groups from RubricNet's training
variance, but over three seeds so the spread is comparable to everything else.
"""

import argparse
import json
import sys
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, ".")

from guitar.prepare_splits import (
    ALL_FEATURES_V5_PRUNED_COLLINEAR2,
    FEATURE_GROUPS_V5_PRUNED_COLLINEAR2 as GROUPS,
    make_piece_id,
)
from guitar.unified_table import METRICS, score_all

DIMS = ("lh", "rh", "global")
PRETTY = {"lh": "left hand", "rh": "right hand", "global": "global"}


def build_sets():
    full = list(ALL_FEATURES_V5_PRUNED_COLLINEAR2)
    sets = {"all (25)": full}
    for d in DIMS:
        sets[f"{PRETTY[d]} only"] = list(GROUPS[d])
    for d in DIMS:
        sets[f"all minus {PRETTY[d]}"] = [f for f in full if f not in GROUPS[d]]
    return sets


def run(features, splits, columns, seeds):
    runs = []
    for split_idx in range(5):
        tr = splits[str(split_idx)]["train"]
        va = splits[str(split_idx)]["val"]
        te = splits[str(split_idx)]["test"]
        pool_ids = list(tr) + list(va)
        X_pool = features.loc[pool_ids, columns]
        y_pool = pd.Series([{**tr, **va}[i] for i in pool_ids], index=pool_ids)
        X_test = features.loc[list(te), columns]
        y_test = pd.Series([te[i] for i in te], index=list(te))

        medians = X_pool.median().fillna(0.0)
        X_pool = X_pool.fillna(medians)
        X_test = X_test.fillna(medians)

        for seed in seeds:
            model = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X_pool, y_pool)
            runs.append(score_all(y_test.to_numpy(), model.predict(X_test)))

    return {key: {"mean": float(mean(m[key] for m in runs)),
                  "std": float(stdev([m[key] for m in runs]))}
            for key, _, _ in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features/guitar_descriptors_v5.csv")
    ap.add_argument("--splits", default="guitar/guitar_splits_v5.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="guitar/dimension_loo_results.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    df = pd.read_csv(args.csv)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    features = df.set_index("piece_id")
    with open(args.splits) as f:
        splits = json.load(f)

    sets = build_sets()
    results = {}
    show = ["accuracy", "balanced_accuracy", "mae", "mse", "macro_mae", "kendall_tau_b"]
    hdr = f"{'feature set':<26}{'#':>4}" + "".join(f"{k[:9]:>12}" for k in show)
    print(hdr)
    print("-" * len(hdr))
    for name, columns in sets.items():
        m = run(features, splits, columns, seeds)
        results[name] = {"n_features": len(columns), **m}
        row = f"{name:<26}{len(columns):>4}"
        for k in show:
            row += f"{m[k]['mean']:>8.4f}±{m[k]['std']:<3.2f}"
        print(row)

    full = results["all (25)"]
    print("\nCost of removing each dimension from the full set "
          "(positive = removing it made the model worse):")
    for d in DIMS:
        r = results[f"all minus {PRETTY[d]}"]
        print(f"  drop {PRETTY[d]:<12} ({25 - len(GROUPS[d])} left): "
              f"dAcc {full['accuracy']['mean'] - r['accuracy']['mean']:+.4f}   "
              f"dMAE {r['mae']['mean'] - full['mae']['mean']:+.4f}   "
              f"dTau {full['kendall_tau_b']['mean'] - r['kendall_tau_b']['mean']:+.4f}   "
              f"(full acc std {full['accuracy']['std']:.4f})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
