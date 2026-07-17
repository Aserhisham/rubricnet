"""Fuzzy rule-based classifiers for guitar difficulty estimation: complete
search of primitive rules and fuzzy pattern trees (see guitar/fuzzy_rules.py
and fuzzy.txt for the underlying paper). Both are deterministic, so unlike
RubricNet's 3-seed evaluation, they are run once per fold.

Protocol mirrors guitar/thesis_extra_results.py and guitar/baselines.py:
frozen splits from guitar/guitar_splits.json, train-fold median imputation,
V3 features from features/guitar_descriptors_v3.csv. Hyperparameters (m for
complete search, d_max for FPT) are selected per fold on the val split by
balanced accuracy, then the model is refit on train+val before scoring test.

Outputs:
- guitar/fuzzy_results_v3.json: per-fold metrics + mean/std summary.
- guitar/fuzzy_rules_dump_v3.json: human-readable rules/trees for the
  qualitative analysis (top-10 complete-search rules and the FPT expression
  per class, per fold).
"""
import argparse
import json
import os
import sys
from statistics import mean, stdev

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error,
)

from guitar.fuzzy_rules import CompleteSearchClassifier, Fuzzifier, FuzzyPatternTreeClassifier
from guitar.prepare_splits import ALL_FEATURES_V3, NUM_CLASSES, make_piece_id

N_SPLITS = 5
D_MAX_GRID = [1, 2, 3, 4, 5]


def fold_xy(features, fold, subset):
    ids = list(fold[subset].keys())
    X = features.loc[ids]
    y = np.array([fold[subset][i] for i in ids])
    return X, y


def accuracy_plus_minus_1(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def compute_metrics(y_true, y_pred):
    res = kendalltau(y_true, y_pred)
    tau = res.correlation if hasattr(res, "correlation") else res[0]
    if np.isnan(tau):
        tau = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "acc_plus_minus_1": accuracy_plus_minus_1(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "kendall_tau": float(tau),
    }


def summarize(per_fold, keys):
    return {
        key: {"mean": mean(vals), "std": stdev(vals)}
        for key in keys
        for vals in [[f[key] for f in per_fold]]
    }


def run_complete_search(M_train, y_train, M_val, y_val, M_trainval, y_trainval, M_test, feature_names):
    val_scores, _ = CompleteSearchClassifier.sweep_m(M_train, y_train, M_val, y_val, n_classes=NUM_CLASSES)
    best_m = max(val_scores, key=lambda m: (val_scores[m], -m))

    clf = CompleteSearchClassifier(m=best_m, n_classes=NUM_CLASSES).fit(M_trainval, y_trainval)
    y_pred = clf.predict(M_test)

    rules_dump = {
        str(c): clf.rules_for_class(c, top_k=10, feature_names=feature_names)
        for c in range(NUM_CLASSES)
    }
    return y_pred, best_m, rules_dump


def run_fpt(M_train, y_train, M_val, y_val, M_trainval, y_trainval, M_test, feature_names, gamma, use_negation, balanced_rmse):
    best_dmax, best_score = None, -1.0
    for d_max in D_MAX_GRID:
        clf = FuzzyPatternTreeClassifier(
            d_max=d_max, gamma=gamma, use_negation=use_negation, balanced_rmse=balanced_rmse, n_classes=NUM_CLASSES,
        ).fit(M_train, y_train)
        score = balanced_accuracy_score(y_val, clf.predict(M_val))
        if score > best_score:
            best_score, best_dmax = score, d_max

    clf = FuzzyPatternTreeClassifier(
        d_max=best_dmax, gamma=gamma, use_negation=use_negation, balanced_rmse=balanced_rmse, n_classes=NUM_CLASSES,
    ).fit(M_trainval, y_trainval)
    y_pred = clf.predict(M_test)

    if clf.is_constant(M_test):
        print(f"  WARNING: FPT has a constant class tree on test (d_max={best_dmax})")

    tree_dump = {
        str(c): {
            "expression": clf.tree_expression(c, feature_names),
            "tree": clf.tree_dict(c, feature_names),
        }
        for c in range(NUM_CLASSES)
    }
    return y_pred, best_dmax, tree_dump, clf.negation_used_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", choices=["cdf", "minmax"], default="cdf")
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--plain-rmse", action="store_true", help="Use unweighted RMSE for FPT instead of class-balanced RMSE")
    parser.add_argument("--no-negation", action="store_true", help="Disable negated leaf statements in FPT")
    parser.add_argument("--v5", action="store_true", help="Use V5 dataset (V3 features, 76 pdf/no-rhythm dummy pieces dropped)")
    parser.add_argument("--out", default="guitar/fuzzy_results_v3.json")
    parser.add_argument("--dump", default="guitar/fuzzy_rules_dump_v3.json")
    args = parser.parse_args()

    balanced_rmse = not args.plain_rmse
    use_negation = not args.no_negation

    if args.v5:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        if args.out == "guitar/fuzzy_results_v3.json":
            args.out = "guitar/fuzzy_results_v5.json"
        if args.dump == "guitar/fuzzy_rules_dump_v3.json":
            args.dump = "guitar/fuzzy_rules_dump_v5.json"
        print("Running fuzzy baselines on V5 dataset (V3 features, no-rhythm pdf pieces dropped)...")
    else:
        csv_path = "features/guitar_descriptors_v3.csv"
        splits_path = "guitar/guitar_splits.json"

    df = pd.read_csv(csv_path)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    features = df.set_index("piece_id")[ALL_FEATURES_V3]
    with open(splits_path) as f:
        splits = json.load(f)

    cs_per_fold, fpt_per_fold = [], []
    cs_rules_by_fold, fpt_trees_by_fold = {}, {}
    any_negation_used = False

    for split_idx in range(N_SPLITS):
        print(f"=== Fold {split_idx} ===")
        fold = splits[str(split_idx)]
        X_train, y_train = fold_xy(features, fold, "train")
        X_val, y_val = fold_xy(features, fold, "val")
        X_test, y_test = fold_xy(features, fold, "test")

        medians = X_train.median().fillna(0.0)
        X_train, X_val, X_test = (x.fillna(medians) for x in (X_train, X_val, X_test))
        X_trainval = pd.concat([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        fz = Fuzzifier(method=args.norm).fit(X_train.values)
        M_train = fz.transform(X_train.values)
        M_val = fz.transform(X_val.values)

        fz_trainval = Fuzzifier(method=args.norm).fit(X_trainval.values)
        M_trainval = fz_trainval.transform(X_trainval.values)
        M_test = fz_trainval.transform(X_test.values)

        y_pred_cs, best_m, rules_dump = run_complete_search(
            M_train, y_train, M_val, y_val, M_trainval, y_trainval, M_test, ALL_FEATURES_V3,
        )
        cs_metrics = compute_metrics(y_test, y_pred_cs)
        cs_metrics["selected_m"] = best_m
        cs_per_fold.append(cs_metrics)
        cs_rules_by_fold[str(split_idx)] = rules_dump
        print(f"  CompleteSearch: m={best_m} acc={cs_metrics['accuracy']:.4f} bacc={cs_metrics['balanced_accuracy']:.4f} "
              f"MAE={cs_metrics['mae']:.4f} tau={cs_metrics['kendall_tau']:.4f}")

        y_pred_fpt, best_dmax, tree_dump, neg_used = run_fpt(
            M_train, y_train, M_val, y_val, M_trainval, y_trainval, M_test, ALL_FEATURES_V3,
            args.gamma, use_negation, balanced_rmse,
        )
        any_negation_used = any_negation_used or neg_used
        fpt_metrics = compute_metrics(y_test, y_pred_fpt)
        fpt_metrics["selected_dmax"] = best_dmax
        fpt_per_fold.append(fpt_metrics)
        fpt_trees_by_fold[str(split_idx)] = tree_dump
        print(f"  FuzzyPatternTree: d_max={best_dmax} acc={fpt_metrics['accuracy']:.4f} bacc={fpt_metrics['balanced_accuracy']:.4f} "
              f"MAE={fpt_metrics['mae']:.4f} tau={fpt_metrics['kendall_tau']:.4f}")

    metric_keys = ["accuracy", "balanced_accuracy", "acc_plus_minus_1", "mae", "mse", "kendall_tau"]
    results = {
        "complete_search": {
            "per_fold": cs_per_fold,
            "summary": summarize(cs_per_fold, metric_keys),
        },
        "fuzzy_pattern_tree": {
            "per_fold": fpt_per_fold,
            "summary": summarize(fpt_per_fold, metric_keys),
        },
        "config": {
            "norm": args.norm,
            "gamma": args.gamma,
            "balanced_rmse": balanced_rmse,
            "negation": use_negation,
            "any_negation_used": any_negation_used,
        },
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    with open(args.dump, "w") as f:
        json.dump({"complete_search_rules": cs_rules_by_fold, "fuzzy_pattern_trees": fpt_trees_by_fold}, f, indent=2)
    print(f"Wrote {args.dump}")

    print("\n=== Summary ===")
    for name, block in [("Complete Search", results["complete_search"]), ("Fuzzy Pattern Tree", results["fuzzy_pattern_tree"])]:
        s = block["summary"]
        print(f"{name}: acc={s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f} "
              f"bacc={s['balanced_accuracy']['mean']:.4f}±{s['balanced_accuracy']['std']:.4f} "
              f"acc±1={s['acc_plus_minus_1']['mean']:.4f}±{s['acc_plus_minus_1']['std']:.4f} "
              f"MAE={s['mae']['mean']:.4f}±{s['mae']['std']:.4f} "
              f"MSE={s['mse']['mean']:.4f}±{s['mse']['std']:.4f} "
              f"tau={s['kendall_tau']['mean']:.4f}±{s['kendall_tau']['std']:.4f}")


if __name__ == "__main__":
    main()
