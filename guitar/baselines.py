"""
Phase 2 baselines for guitar difficulty classification.

Two baselines evaluated across the 5 fixed folds in `guitar/guitar_splits.json`:
- Ordinal regression, reusing `rubricnet.rubricnet.LogisticRegressionOrdinal`
  (via `RubricnetSklearn`) directly against the guitar descriptors.
- Random Forest / Decision Tree, with feature importances.

Reports accuracy / balanced accuracy / MAE / MSE per fold and averaged, and
writes the full results to `guitar/baseline_results.json` or `baseline_results_v2.json`.
"""
import os
import sys
import argparse

os.environ.setdefault("WANDB_MODE", "disabled")

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from guitar.prepare_splits import (
    ALL_FEATURES, ALL_FEATURES_V2, ALL_FEATURES_V3, ALL_FEATURES_V3_PRUNED,
    ALL_FEATURES_V3_BASE, ALL_FEATURES_V3_BASE_NEW, ALL_FEATURES_V5_PRUNED,
    ALL_FEATURES_V5_PRUNED_COLLINEAR, ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW,
    ALL_FEATURES_V5_PRUNED_COLLINEAR2,
    NUM_CLASSES, make_piece_id,
)
from rubricnet.rubricnet import RubricnetSklearn

N_SPLITS = 5

ORDINAL_ARGS = dict(
    lr=0.005,
    batch_size=16,
    hidden_size=32,
    num_layers=1,
    dropout=0.05,
    decay_lr=0.5,
    weight_decay=1e-4,
    patience=20,
    alias_experiment="guitar_baseline_ordinal",
)


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_data(csv_path="features/guitar_descriptors.csv", splits_path="guitar/guitar_splits.json", columns=ALL_FEATURES):
    df = pd.read_csv(csv_path)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    features = df.set_index("piece_id")[columns]
    with open(splits_path) as f:
        splits = json.load(f)
    return features, splits


def get_fold_xy(features, splits, split_idx, subset):
    fold_labels = splits[str(split_idx)][subset]
    ids = list(fold_labels.keys())
    X = features.loc[ids]
    y = pd.Series([fold_labels[i] for i in ids], index=ids)
    return X, y


def score(y_true, y_pred):
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        mse=mean_squared_error(y_true, y_pred),
    )


def summarize(name, fold_scores):
    print(f"\n{name}")
    metrics = {}
    for key in ("accuracy", "balanced_accuracy", "mae", "mse"):
        values = [s[key] for s in fold_scores]
        metrics[key] = values
        print(f"  {key:18s} {mean(values):.4f} +/- {stdev(values):.4f}")
    return metrics


def run_ordinal_regression(features, splits, columns, alias_experiment, num_classes=NUM_CLASSES):
    fold_scores = []
    for split_idx in range(N_SPLITS):
        X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
        X_val, y_val = get_fold_xy(features, splits, split_idx, "val")
        X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

        # Train-fold median imputation
        medians = X_train.median().fillna(0.0)
        X_train = X_train.fillna(medians)
        X_val = X_val.fillna(medians)
        X_test = X_test.fillna(medians)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        opt_args = ORDINAL_ARGS.copy()
        opt_args["alias_experiment"] = alias_experiment
        args = Args(**opt_args)
        
        clf = RubricnetSklearn(
            input_dim=len(columns), num_classes=num_classes, split=split_idx, args=args, logging=False
        )
        clf.fit(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)
        clf.load_model(f"checkpoints/{args.alias_experiment}/split_{split_idx}.ckpt")

        y_pred = clf.predict(X_test_s).cpu().numpy()
        fold_scores.append(score(y_test, y_pred))
        print(f"  split {split_idx}: acc={fold_scores[-1]['accuracy']:.4f} MAE={fold_scores[-1]['mae']:.4f}")

    metrics = summarize("Ordinal regression (LogisticRegressionOrdinal)", fold_scores)
    return metrics


def run_tree_baseline(features, splits, columns, model_cls, name, **model_kwargs):
    fold_scores = []
    importances = []
    for split_idx in range(N_SPLITS):
        X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
        X_val, y_val = get_fold_xy(features, splits, split_idx, "val")
        X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

        # Train-fold median imputation
        medians = X_train.median().fillna(0.0)
        X_train = X_train.fillna(medians)
        X_val = X_val.fillna(medians)
        X_test = X_test.fillna(medians)

        # no early stopping needed here, so fold val back into train
        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])

        model = model_cls(random_state=42, **model_kwargs)
        model.fit(X_trainval, y_trainval)
        y_pred = model.predict(X_test)

        fold_scores.append(score(y_test, y_pred))
        importances.append(model.feature_importances_)
        print(f"  split {split_idx}: acc={fold_scores[-1]['accuracy']:.4f} MAE={fold_scores[-1]['mae']:.4f}")

    metrics = summarize(name, fold_scores)

    mean_importance = np.mean(importances, axis=0)
    ranked = sorted(zip(columns, mean_importance), key=lambda t: -t[1])
    print("  feature importances (mean across folds):")
    for feat, imp in ranked:
        print(f"    {feat:28s} {imp:.4f}")
    metrics["feature_importances"] = {feat: float(imp) for feat, imp in ranked}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Use version 2 features")
    parser.add_argument("--v3", action="store_true", help="Use version 3 features")
    parser.add_argument("--v3-pruned", action="store_true", help="Use version 3 features minus near-zero-signal descriptors")
    parser.add_argument("--v3-base", action="store_true", help="Use V3 base features (unified provenance, no rhythm-aware features)")
    parser.add_argument("--v3-base-new", action="store_true", help="Use V3 base + hand-crafted interaction features")
    parser.add_argument("--v5", action="store_true", help="Use V5 dataset (V3 features, 76 pdf/no-rhythm dummy pieces dropped)")
    parser.add_argument("--v6", action="store_true", help="Paper-comparison run: V5 pieces, 9 equal-width classes, 60/20/20 splits (see prepare_splits_v6.py)")
    parser.add_argument("--v5-pruned", action="store_true", help="V5 dataset minus avg_polyphony/fret_change_rate (expert review, see guitar/EXPERT_MEETING.md)")
    parser.add_argument("--v5-pruned-collinear-expert-new", action="store_true", help="V5-pruned-collinear (27 feat) + 7 new expert-proposed descriptors -- decide on VAL accuracy, not test")
    parser.add_argument("--v5-pruned-collinear", action="store_true", help="V5-pruned (30 feat) minus high_position_ratio/avg_fret/p90_fret (collinear w/ fret_entropy, expert review)")
    parser.add_argument("--v5-pruned-collinear2", action="store_true", help="V5-pruned-collinear (27 feat) minus p90_stretch_velocity_beats/p95_position_shift_window -- decide on VAL accuracy, not test")
    args = parser.parse_args()

    num_classes = 9 if args.v6 else NUM_CLASSES
    splits_path = "guitar/guitar_splits.json"
    if args.v5_pruned_collinear:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR
        alias = "guitar_baseline_ordinal_v5_pruned_collinear"
        out_path = "guitar/baseline_results_v5_pruned_collinear.json"
        print("Running baselines on V5 pruned-collinear (27 feat, first collinear-cluster pass)...")
    elif args.v5_pruned_collinear2:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR2
        alias = "guitar_baseline_ordinal_v5_pruned_collinear2"
        out_path = "guitar/baseline_results_v5_pruned_collinear2.json"
        print("Running baselines on V5 pruned-collinear2 (25 feat, second collinear-cluster pass)...")
    elif args.v5_pruned_collinear_expert_new:
        csv_path = "features/guitar_descriptors_v5_expert_new.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW
        alias = "guitar_baseline_ordinal_v5_pruned_collinear_expert_new"
        out_path = "guitar/baseline_results_v5_pruned_collinear_expert_new.json"
        print("Running baselines on V5 pruned-collinear + 7 new expert-proposed descriptors...")
    elif args.v5_pruned:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED
        alias = "guitar_baseline_ordinal_v5_pruned"
        out_path = "guitar/baseline_results_v5_pruned.json"
        print("Running baselines on V5 pruned (avg_polyphony/fret_change_rate dropped, expert review)...")
    elif args.v6:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v6.json"
        columns = ALL_FEATURES_V3
        alias = "guitar_baseline_ordinal_v6"
        out_path = "guitar/baseline_results_v6.json"
        print("Running baselines on V6 (paper-comparison: 9 equal-width classes, 60/20/20 splits)...")
    elif args.v5:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V3
        alias = "guitar_baseline_ordinal_v5"
        out_path = "guitar/baseline_results_v5.json"
        print("Running baselines on V5 dataset (V3 features, no-rhythm pdf pieces dropped)...")
    elif args.v3_base_new:
        csv_path = "features/guitar_descriptors_v3_base.csv"
        columns = ALL_FEATURES_V3_BASE_NEW
        alias = "guitar_baseline_ordinal_v3_base_new"
        out_path = "guitar/baseline_results_v3_base_new.json"
        print("Running baselines on V3 base + interaction features...")
    elif args.v3_base:
        csv_path = "features/guitar_descriptors_v3.csv"
        columns = ALL_FEATURES_V3_BASE
        alias = "guitar_baseline_ordinal_v3_base"
        out_path = "guitar/baseline_results_v3_base.json"
        print("Running baselines on V3 base features (no rhythm)...")
    elif args.v3_pruned:
        csv_path = "features/guitar_descriptors_v3.csv"
        columns = ALL_FEATURES_V3_PRUNED
        alias = "guitar_baseline_ordinal_v3_pruned"
        out_path = "guitar/baseline_results_v3_pruned.json"
        print("Running baselines on V3 pruned features...")
    elif args.v3:
        csv_path = "features/guitar_descriptors_v3.csv"
        columns = ALL_FEATURES_V3
        alias = "guitar_baseline_ordinal_v3"
        out_path = "guitar/baseline_results_v3.json"
        print("Running baselines on V3 features...")
    elif args.v2:
        csv_path = "features/guitar_descriptors_v2.csv"
        columns = ALL_FEATURES_V2
        alias = "guitar_baseline_ordinal_v2"
        out_path = "guitar/baseline_results_v2.json"
        print("Running baselines on V2 features...")
    else:
        csv_path = "features/guitar_descriptors.csv"
        columns = ALL_FEATURES
        alias = "guitar_baseline_ordinal"
        out_path = "guitar/baseline_results.json"
        print("Running baselines on V1 features...")

    features, splits = load_data(csv_path=csv_path, splits_path=splits_path, columns=columns)

    results = {
        "ordinal_regression": run_ordinal_regression(features, splits, columns, alias, num_classes=num_classes),
        "random_forest": run_tree_baseline(features, splits, columns, RandomForestClassifier, "Random Forest", n_estimators=200),
        "decision_tree": run_tree_baseline(features, splits, columns, DecisionTreeClassifier, "Decision Tree", max_depth=6),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
