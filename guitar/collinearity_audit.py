"""Multicollinearity audit of the adopted 25-descriptor set.

Why this exists
---------------
Section 4.3.4 prunes descriptor pairs at Pearson |r| > 0.9, arguing that because each
RubricNet subnetwork is a plain affine map, correlation above that bar is "classic
linear-regression multicollinearity". That argument is right, but a pairwise |r|
threshold is the wrong instrument for it: multicollinearity is a property of the whole
design matrix, and the standard diagnostic is the variance inflation factor. This
script runs the diagnostic the argument implies, over the set that was actually adopted.

Outputs the VIF for every descriptor, the condition number of the correlation matrix,
and the strongest pairwise relationships under both Pearson (linear redundancy, what
the pruning tested) and Spearman (rank redundancy, which a monotone transform of an
existing descriptor leaves at exactly 1.0 while Pearson stays well below the bar).
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")

from guitar.prepare_splits import (
    ALL_FEATURES_V5_PRUNED_COLLINEAR2,
    FEATURE_GROUPS_V5_PRUNED_COLLINEAR2,
    make_piece_id,
)

VIF_WARN = 10.0      # conventional cutoff
PEARSON_BAR = 0.90   # the bar section 4.3.4 actually applied


def load_matrix(csv_path, columns):
    df = pd.read_csv(csv_path)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    X = df.set_index("piece_id")[list(columns)].astype(float)
    return X.fillna(X.median())


def vif_table(X):
    Z = (X - X.mean()) / X.std()
    R = np.corrcoef(Z.to_numpy().T)
    vif = np.diag(np.linalg.inv(R))
    cond = float(np.linalg.cond(R))
    return pd.Series(vif, index=X.columns).sort_values(ascending=False), cond


def pair_table(X, method):
    cols = list(X.columns)
    if method == "pearson":
        C = np.abs(X.corr().to_numpy())
    else:
        C = np.abs(spearmanr(X.to_numpy()).statistic)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append((float(C[i, j]), cols[i], cols[j]))
    rows.sort(reverse=True)
    return rows


def group_of(name):
    for g, feats in FEATURE_GROUPS_V5_PRUNED_COLLINEAR2.items():
        if name in feats:
            return g
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features/guitar_descriptors_v5.csv")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="guitar/collinearity_audit.json")
    args = ap.parse_args()

    columns = ALL_FEATURES_V5_PRUNED_COLLINEAR2
    X = load_matrix(args.csv, columns)
    print(f"{len(columns)} descriptors, {len(X)} pieces\n")

    vif, cond = vif_table(X)
    print("Variance inflation factors (cutoff 10):")
    for name, v in vif.items():
        flag = "  <-- above cutoff" if v > VIF_WARN else ""
        print(f"  {v:8.2f}  {name:38s} [{group_of(name)}]{flag}")
    print(f"\nCondition number of the correlation matrix: {cond:.1f}")

    pearson = pair_table(X, "pearson")
    spearman = pair_table(X, "spearman")

    print(f"\nTop {args.top} pairs by |Pearson r|  (the pruning bar was {PEARSON_BAR}):")
    for v, a, b in pearson[: args.top]:
        flag = "  <-- above the pruning bar" if v > PEARSON_BAR else ""
        print(f"  {v:.3f}  {a} / {b}{flag}")

    print(f"\nTop {args.top} pairs by |Spearman rho|:")
    for v, a, b in spearman[: args.top]:
        flag = "  <-- rank-identical" if v > 0.999 else ""
        print(f"  {v:.3f}  {a} / {b}{flag}")

    survivors = [(v, a, b) for v, a, b in pearson if v > PEARSON_BAR]
    print(f"\nPairs still above the {PEARSON_BAR} Pearson bar after pruning: {len(survivors)}")
    rank_identical = [(v, a, b) for v, a, b in spearman if v > 0.999]
    print(f"Rank-identical pairs (Spearman = 1.0): {len(rank_identical)}")
    for v, a, b in rank_identical:
        pear = next(p for p, x, y in pearson if {x, y} == {a, b})
        print(f"  {a} / {b}: Spearman {v:.4f} but Pearson {pear:.3f} "
              f"-- invisible to a Pearson-only audit")

    with open(args.out, "w") as f:
        json.dump({
            "n_descriptors": len(columns),
            "vif": {k: float(v) for k, v in vif.items()},
            "condition_number": cond,
            "top_pearson": [{"r": v, "a": a, "b": b} for v, a, b in pearson[: args.top]],
            "top_spearman": [{"rho": v, "a": a, "b": b} for v, a, b in spearman[: args.top]],
            "above_vif_cutoff": [k for k, v in vif.items() if v > VIF_WARN],
            "rank_identical_pairs": [{"a": a, "b": b} for _, a, b in rank_identical],
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
