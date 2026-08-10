"""
Isolate pure random-seed variance for the CIPI reproduction: hyperparameters
are held completely FIXED (the paper's assumed defaults), and only the random
seed changes across trials -- unlike retune_cipi_paper.py, which varied both
hyperparameters AND (implicitly) seed together. This answers: does seed noise
alone, at a single fixed good hyperparameter config, explain reaching 41.4%?

Uses the pristine rubricnet_original code (same as reproduce_cipi_paper_original.py).
Each seed trains all 5 folds with patience=30 (fast proxy, same tradeoff as the
hyperparameter search) -- not the full patience=400 final-number regime.

Usage:
    python seed_sweep_cipi_paper.py --n-seeds 30
"""
import json
import os
import sys
from statistics import mean, stdev

import lightning.pytorch as pl
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubricnet_original.rubricnet import RubricnetSklearn

SPLITS_PATH = "rubricnet/cipi_splits.json"
FEATURES_PATH = "features/cipi-features-ISMIR24.json"
NUM_CLASSES = 9

BASIC_FEATURES = [
    'rh_pitch_set_lz',
    'lh_pitch_set_lz',
    'rh_pitch_range',
    'lh_pitch_range',
    'lh_average_pitch',
    'rh_average_pitch',
    'lh_average_ioi_seconds',
    'rh_average_ioi_seconds',
    'rh_displacement_rate',
    'lh_displacement_rate',
    'lh_pitch_entropy',
    'rh_pitch_entropy',
]


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_mse_macro(y_true, y_pred):
    mse_each_class = []
    for true_class in set(y_true):
        tt, pp = zip(*[(t, p) for t, p in zip(y_true, y_pred) if t == true_class])
        mse_each_class.append(mean_squared_error(y_true=tt, y_pred=pp))
    return mean(mse_each_class)


def run_5fold(args, alias, splits, features, seed):
    pl.seed_everything(seed, workers=True)
    acc9_test, mse_test = [], []
    for split in range(5):
        ids_train = list(splits[str(split)]["train"].keys())
        ids_val = list(splits[str(split)]["val"].keys())
        ids_test = list(splits[str(split)]["test"].keys())
        y_train = pd.Series(splits[str(split)]["train"].values())
        y_val = pd.Series(splits[str(split)]["val"].values())
        y_test = pd.Series(splits[str(split)]["test"].values())

        from sklearn import preprocessing
        X_train = pd.DataFrame({ft: [features[i][ft] for i in ids_train] for ft in BASIC_FEATURES})
        X_val = pd.DataFrame({ft: [features[i][ft] for i in ids_val] for ft in BASIC_FEATURES})
        X_test = pd.DataFrame({ft: [features[i][ft] for i in ids_test] for ft in BASIC_FEATURES})
        scaler = preprocessing.StandardScaler().fit(X_train)

        clf = RubricnetSklearn(
            input_dim=len(BASIC_FEATURES), num_classes=NUM_CLASSES, split=split, args=args, logging=False
        )
        clf.fit(
            scaler.transform(X_train), y_train,
            scaler.transform(X_val), y_val,
            scaler.transform(X_test), y_test,
        )
        clf.load_model(f"checkpoints/{alias}/split_{split}.ckpt")
        pred_test = clf.predict(scaler.transform(X_test))

        acc9_test.append(balanced_accuracy_score(y_true=y_test, y_pred=pred_test))
        mse_test.append(get_mse_macro(y_true=y_test, y_pred=pred_test))
    return mean(acc9_test), mean(mse_test)


def main(n_seeds):
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    with open(FEATURES_PATH) as f:
        features = {row["id"]: row for row in json.load(f)}

    # Paper's assumed default hyperparameters -- FIXED across all seeds.
    args = Args(
        batch_size=40, patience=30, lr=0.1, hidden_size=1, num_layers=1,
        dropout=0.1, decay_lr=0.9, weight_decay=0.01, alias_experiment="seed_sweep",
    )

    results = []
    for seed in range(n_seeds):
        acc9, mse = run_5fold(args, f"seed_sweep_{seed}", splits, features, seed)
        results.append({"seed": seed, "acc9": acc9, "mse": mse})
        print(f"[seed {seed}] acc9={acc9:.4f} mse={mse:.4f}")

    accs = [r["acc9"] for r in results]
    print("\n=== Seed sweep, FIXED hyperparameters, patience=30 ===")
    print(f"n={len(accs)}  mean={mean(accs)*100:.1f}  std={stdev(accs)*100:.2f}  min={min(accs)*100:.1f}  max={max(accs)*100:.1f}")
    n_hit = sum(1 for a in accs if a >= 0.414)
    print(f"seeds reaching >=41.4%: {n_hit}/{len(accs)}")

    with open("seed_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    n = 30
    if "--n-seeds" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n-seeds") + 1])
    main(n)
