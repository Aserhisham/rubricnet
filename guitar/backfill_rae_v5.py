"""
Backfill per-class RAE onto the finished V5 RubricNet run, using the
checkpoints already saved under checkpoints/guitar_rubricnet_final_v5_seed_{0,1,2}/
split_{0-4}.ckpt -- pure inference, no retraining.

The StandardScaler used at train time was never persisted to disk, but its
.fit() is deterministic given the same train fold (median-imputed the same
way), so refitting it here reproduces the exact scaler used originally.
"""
import json
import os
import sys
from statistics import mean, stdev

import numpy as np

os.environ.setdefault("WANDB_MODE", "disabled")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V3, NUM_CLASSES
from guitar.train_guitar_rubricnet import DEFAULT_HYPERPARAMS, relative_absolute_error_per_class
from rubricnet.rubricnet import RubricnetSklearn

CSV_PATH = "features/guitar_descriptors_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
ALIAS_EXPERIMENT = "guitar_rubricnet_final_v5"
BEST_HYPERPARAMS_PATH = "guitar/best_hyperparams_guitar_all_v5.json"
RESULTS_PATH = "guitar/rubricnet_results_v5.json"
CKPT_DIR = "checkpoints"


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
    features, splits = load_data(csv_path=CSV_PATH, splits_path=SPLITS_PATH, columns=ALL_FEATURES_V3)
    hyperparams = load_hyperparams(BEST_HYPERPARAMS_PATH)

    seeds = [0, 1, 2]
    rae_run = []

    for seed in seeds:
        rae_seed = []
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
                input_dim=len(ALL_FEATURES_V3), num_classes=NUM_CLASSES,
                split=split_idx, args=args_cls, logging=False,
            )
            clf.load_model(f"{CKPT_DIR}/{alias}/split_{split_idx}.ckpt")

            y_pred = clf.predict(scaler.transform(X_test)).cpu().numpy()
            y_pred = np.clip(y_pred, 0, NUM_CLASSES - 1)

            rae_total, rae_per_class = relative_absolute_error_per_class(y_test.to_numpy(), y_pred)
            rae_seed.append(rae_total)
            per_class_str = {int(k): round(v, 4) for k, v in rae_per_class.items()}
            print(f"  seed {seed} split {split_idx}: RAE={rae_total:.4f}  per-class={per_class_str}")
        rae_run.append(rae_seed)

    flat = np.array(rae_run).flatten()
    print(f"\nV5 RAE: {mean(flat):.4f} +/- {stdev(flat):.4f}")

    with open(RESULTS_PATH) as f:
        results = json.load(f)
    results["metrics"]["rae"] = {
        "per_fold_per_seed": rae_run,
        "mean": float(mean(flat)),
        "std": float(stdev(flat)),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote RAE back into {RESULTS_PATH}")


if __name__ == "__main__":
    main()
