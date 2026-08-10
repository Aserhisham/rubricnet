"""Reproduces the two result sets computed directly for the thesis text
(thesis/chapters/4_results.tex) that are not covered by baselines.py:

1. Per-dimension ablation (Table 4.2): Random Forest trained on the V3
   left-hand / right-hand / global descriptor groups separately and on the
   full 32-descriptor set.
2. Kendall's tau for the Decision Tree and Random Forest baselines on V2/V3
   (Table 4.1 tau column; baselines.py never computed tau).

Protocol is identical to baselines.py run_tree_baseline: frozen splits from
guitar/guitar_splits.json, train-fold median imputation, train+val refit,
RandomForestClassifier(n_estimators=200) / DecisionTreeClassifier(max_depth=6),
random_state=42. The all-dimensions RF row reproduces the published
Random Forest V3 numbers exactly, which validates the protocol match.

V1 tau is deliberately not reported: the current guitar_descriptors.csv no
longer reproduces the published V1 baseline rows (the CSV was re-extracted
after those runs), so freshly computed V1 tau values would not be consistent
with the published V1 table row.
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from statistics import mean, stdev

import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier

from guitar.prepare_splits import (
    ALL_FEATURES_V2, ALL_FEATURES_V3, ALL_FEATURES_V5_PRUNED_COLLINEAR2,
    ALL_FEATURES_V5_PRUNED_COLLINEAR,
    FEATURE_GROUPS_V3, FEATURE_GROUPS_V5_PRUNED_COLLINEAR2, make_piece_id,
)

N_SPLITS = 5


def fold_xy(features, fold, subset):
    ids = list(fold[subset].keys())
    return features.loc[ids], pd.Series([fold[subset][i] for i in ids], index=ids)


def run_folds(features, splits, model_factory):
    """Same protocol as baselines.py run_tree_baseline; also returns tau."""
    fold_scores = []
    for split_idx in range(N_SPLITS):
        fold = splits[str(split_idx)]
        X_train, y_train = fold_xy(features, fold, "train")
        X_val, y_val = fold_xy(features, fold, "val")
        X_test, y_test = fold_xy(features, fold, "test")

        medians = X_train.median().fillna(0.0)
        X_train, X_val, X_test = (x.fillna(medians) for x in (X_train, X_val, X_test))

        model = model_factory()
        model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        y_pred = model.predict(X_test)
        fold_scores.append(dict(
            accuracy=accuracy_score(y_test, y_pred),
            balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
            mae=mean_absolute_error(y_test, y_pred),
            mse=mean_squared_error(y_test, y_pred),
            kendall_tau=kendalltau(y_test, y_pred).statistic,
        ))
    return {
        key: {"mean": mean(vals), "std": stdev(vals)}
        for key in fold_scores[0]
        for vals in [[s[key] for s in fold_scores]]
    }


def main():
    with open("guitar/guitar_splits.json") as f:
        splits = json.load(f)
    with open("guitar/guitar_splits_v5.json") as f:
        splits_v5 = json.load(f)

    def rf():
        return RandomForestClassifier(random_state=42, n_estimators=200)

    def dt():
        return DecisionTreeClassifier(random_state=42, max_depth=6)

    print("=== Per-dimension ablation (RF, V3 descriptors) — thesis Table 4.2 ===")
    df3 = pd.read_csv("features/guitar_descriptors_v3.csv")
    df3["piece_id"] = df3.apply(make_piece_id, axis=1)
    feature_sets = {
        "lh_only": FEATURE_GROUPS_V3["lh"],
        "rh_only": FEATURE_GROUPS_V3["rh"],
        "global_only": FEATURE_GROUPS_V3["global"],
        "all_v3": ALL_FEATURES_V3,
    }
    results = {"dimension_ablation": {}, "baseline_kendall_tau": {}, "dimension_ablation_v5_pruned_collinear2": {}}
    for name, columns in feature_sets.items():
        m = run_folds(df3.set_index("piece_id")[columns], splits, rf)
        results["dimension_ablation"][name] = {"n_features": len(columns), **m}
        print(f"  {name:12s} ({len(columns):2d}): acc={m['accuracy']['mean']:.4f}±{m['accuracy']['std']:.4f} "
              f"bacc={m['balanced_accuracy']['mean']:.4f}±{m['balanced_accuracy']['std']:.4f} "
              f"MAE={m['mae']['mean']:.4f} MSE={m['mse']['mean']:.4f}")

    print("\n=== Per-dimension ablation (RF, V5-pruned-collinear2, 25 feat) — adopted headline set ===")
    df5 = pd.read_csv("features/guitar_descriptors_v5.csv")
    df5["piece_id"] = df5.apply(make_piece_id, axis=1)
    feature_sets_v5 = {
        "lh_only": FEATURE_GROUPS_V5_PRUNED_COLLINEAR2["lh"],
        "rh_only": FEATURE_GROUPS_V5_PRUNED_COLLINEAR2["rh"],
        "global_only": FEATURE_GROUPS_V5_PRUNED_COLLINEAR2["global"],
        "all_v5_pruned_collinear2": ALL_FEATURES_V5_PRUNED_COLLINEAR2,
    }
    for name, columns in feature_sets_v5.items():
        m = run_folds(df5.set_index("piece_id")[columns], splits_v5, rf)
        results["dimension_ablation_v5_pruned_collinear2"][name] = {"n_features": len(columns), **m}
        print(f"  {name:12s} ({len(columns):2d}): acc={m['accuracy']['mean']:.4f}±{m['accuracy']['std']:.4f} "
              f"bacc={m['balanced_accuracy']['mean']:.4f}±{m['balanced_accuracy']['std']:.4f} "
              f"MAE={m['mae']['mean']:.4f} MSE={m['mse']['mean']:.4f}")

    print("\n=== Tree-baseline Kendall tau (V2/V3/V5*) — thesis Table 5.1 tau column ===")
    for gen, csv_path, columns, use_splits in [
        ("V2", "features/guitar_descriptors_v2.csv", ALL_FEATURES_V2, splits),
        ("V3", "features/guitar_descriptors_v3.csv", ALL_FEATURES_V3, splits),
        # V5 (32 feat) reuses the V3 column list on the 640-piece V5 csv/splits.
        ("V5", "features/guitar_descriptors_v5.csv", ALL_FEATURES_V3, splits_v5),
        ("V5_pruned_collinear", "features/guitar_descriptors_v5.csv", ALL_FEATURES_V5_PRUNED_COLLINEAR, splits_v5),
        ("V5_pruned_collinear2", "features/guitar_descriptors_v5.csv", ALL_FEATURES_V5_PRUNED_COLLINEAR2, splits_v5),
    ]:
        df = pd.read_csv(csv_path)
        df["piece_id"] = df.apply(make_piece_id, axis=1)
        features = df.set_index("piece_id")[columns]
        for model_name, factory in [("random_forest", rf), ("decision_tree", dt)]:
            m = run_folds(features, use_splits, factory)
            results["baseline_kendall_tau"][f"{gen}_{model_name}"] = m
            print(f"  {gen} {model_name:14s}: tau={m['kendall_tau']['mean']:.4f}±{m['kendall_tau']['std']:.4f} "
                  f"(acc check {m['accuracy']['mean']:.4f})")

    out_path = "guitar/thesis_extra_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
