"""
Backfill per-level accuracy + per-level relative absolute error (RAE) onto the
V5-pruned-collinear RubricNet run (current default model, see
guitar/EXPERT_FEATURE_REVIEW.md), using the checkpoints already saved under
checkpoints/guitar_rubricnet_final_v5_pruned_collinear_seed_{0,1,2}/split_{0-4}.ckpt
-- pure inference, no retraining.

For each of the 8 difficulty levels this reports:
  - accuracy: fraction of that level's pieces the model got exactly right
  - RAE: sum|pred-true| / sum|true-global_mean| for that level's pieces, i.e.
    error relative to a "always predict the overall mean" baseline. This is
    the metric requested by the professor: because the level distribution is
    unequal (class sizes 37-136, see prepare_splits.py), a plain per-level MAE
    isn't comparable across levels with different underlying score spread --
    RAE normalizes each level's error against its own baseline difficulty.

Predictions are pooled across all 3 seeds x 5 folds (each seed reuses the same
5 splits, only the model init/training seed differs -- see guitar_splits_v5.json),
so every piece contributes 3 predictions and per-level stats aren't skewed by
small per-fold sample counts.
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("WANDB_MODE", "disabled")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR, NUM_CLASSES
from guitar.train_guitar_rubricnet import (
    DEFAULT_HYPERPARAMS, accuracy_per_class, correct_count_and_distance_per_class,
    relative_absolute_error_per_class,
)
from rubricnet.rubricnet import RubricnetSklearn

CSV_PATH = "features/guitar_descriptors_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
ALIAS_EXPERIMENT = "guitar_rubricnet_final_v5_pruned_collinear"
BEST_HYPERPARAMS_PATH = "guitar/best_hyperparams_guitar_all_v5.json"
RESULTS_PATH = "guitar/rubricnet_results_v5_pruned_collinear.json"
CKPT_DIR = "checkpoints"

# Level ranges per class, from the frozen equal-frequency binning
# (see prepare_splits.py docstring) -- for labeling the table only.
LEVEL_RANGES = ["1-3", "4-5", "6-7", "8", "9-10", "11-12", "13-15", "16-20"]


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_hyperparams(path):
    with open(path) as f:
        tuned = json.load(f)["params"]
    hyperparams = dict(DEFAULT_HYPERPARAMS)
    hyperparams.update(tuned)
    return hyperparams


def main():
    features, splits = load_data(csv_path=CSV_PATH, splits_path=SPLITS_PATH, columns=ALL_FEATURES_V5_PRUNED_COLLINEAR)
    hyperparams = load_hyperparams(BEST_HYPERPARAMS_PATH)

    seeds = [0, 1, 2]
    all_y_true = []
    all_y_pred = []

    for seed in seeds:
        alias = f"{ALIAS_EXPERIMENT}_seed_{seed}"
        for split_idx in range(5):
            X_train, _ = get_fold_xy(features, splits, split_idx, "train")
            X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

            medians = X_train.median().fillna(0.0)
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)

            scaler = StandardScaler().fit(X_train)
            args_cls = Args(alias_experiment=alias, **hyperparams)

            clf = RubricnetSklearn(
                input_dim=len(ALL_FEATURES_V5_PRUNED_COLLINEAR), num_classes=NUM_CLASSES,
                split=split_idx, args=args_cls, logging=False,
            )
            clf.load_model(f"{CKPT_DIR}/{alias}/split_{split_idx}.ckpt")

            y_pred = clf.predict(scaler.transform(X_test)).cpu().numpy()
            y_pred = np.clip(y_pred, 0, NUM_CLASSES - 1)

            all_y_true.append(y_test.to_numpy())
            all_y_pred.append(y_pred)

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    rae_total, rae_per_class = relative_absolute_error_per_class(y_true, y_pred)
    acc_per_class = accuracy_per_class(y_true, y_pred)
    count_dist_per_class = correct_count_and_distance_per_class(y_true, y_pred)

    counts = {int(c): int((y_true == c).sum()) for c in sorted(set(y_true.tolist()))}
    mae_per_class = {
        int(c): float(np.abs(y_pred[y_true == c] - y_true[y_true == c]).mean())
        for c in sorted(counts)
    }

    print(f"\nPooled over {len(seeds)} seeds x 5 folds, N={len(y_true)} predictions "
          f"({len(y_true) // len(seeds)} pieces x {len(seeds)} seeds)\n")
    print("| Level (class) | Level range | N pieces (x3 seeds) | Accuracy | MAE | RAE |")
    print("|---|---|---|---|---|---|")
    for c in sorted(counts):
        print(f"| {c} | {LEVEL_RANGES[c]} | {counts[c]} | {acc_per_class[float(c)]:.3f} | {mae_per_class[c]:.3f} | {rae_per_class[float(c)]:.3f} |")
    print(f"\nOverall: accuracy={float((y_true == y_pred).mean()):.4f}  RAE={rae_total:.4f}")
    print("\nNote: RAE for a class is inflated when that class's true values sit close to the\n"
          "global mean (small baseline denominator) -- compare against MAE, not RAE alone,\n"
          "before concluding a level is a genuine weak point.")

    print("\n--- Third measure: raw correct-piece counts + mean distance from correct level ---\n")
    print("| Level (class) | Level range | Correct / Total | Mean distance (levels off) |")
    print("|---|---|---|---|")
    for c in sorted(counts):
        d = count_dist_per_class[float(c)]
        print(f"| {c} | {LEVEL_RANGES[c]} | {d['n_correct']} / {d['n_total']} | {d['mean_distance']:.3f} |")

    with open(RESULTS_PATH) as f:
        results = json.load(f)
    results["metrics"]["per_level"] = {
        "note": "Pooled across all 3 seeds x 5 folds (each piece contributes 3 predictions, one per seed).",
        "level_ranges": LEVEL_RANGES,
        "n_per_class": counts,
        "accuracy_per_class": {int(k): v for k, v in acc_per_class.items()},
        "mae_per_class": mae_per_class,
        "rae_per_class": {int(k): v for k, v in rae_per_class.items()},
        "correct_count_and_distance_per_class": {int(k): v for k, v in count_dist_per_class.items()},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote per-level accuracy + RAE into {RESULTS_PATH}")


if __name__ == "__main__":
    main()
