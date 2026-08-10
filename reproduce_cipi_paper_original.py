"""
Reproduce the original RubricNet paper (Ramoneda et al., ISMIR 2024) on the
CIPI piano dataset, using the PRISTINE original model code (rubricnet_original/,
pulled from this repo's initial commit before this project's guitar-adaptation
edits) instead of the shared rubricnet/rubricnet.py, which has since been
modified with num_classes>8 special-casing (custom final-layer init, sum-decode,
softened class weights) for the guitar V4/V6 experiments. CIPI is a 9-class
task, so reproduce_cipi_paper.py had been silently running that guitar-specific
code path instead of the paper's original mechanics -- this script fixes that.

Target (paper Table 3/4, "RubricNet proposed" / "Ours", CIPI, basic+LZ
descriptors): Acc-9 = 41.4 (+/-3.1), MSE = 1.7 (+/-0.5).

Run from the repo root:
    python reproduce_cipi_paper_original.py
"""
import json
import os
import sys
from statistics import mean, stdev

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubricnet_original.rubricnet import RubricnetSklearn

SPLITS_PATH = "rubricnet/cipi_splits.json"
FEATURES_PATH = "features/cipi-features-ISMIR24.json"
ALIAS = "cipi_paper_reproduction_original_code"
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


def get_acc1_macro(y_true, y_pred):
    acc_each_class = []
    for true_class in set(y_true):
        matches = [1 if p in (t - 1, t, t + 1) else 0 for t, p in zip(y_true, y_pred) if t == true_class]
        acc_each_class.append(sum(matches) / len(matches))
    return mean(acc_each_class)


def main():
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    with open(FEATURES_PATH) as f:
        features = {row["id"]: row for row in json.load(f)}

    args = Args(
        batch_size=40,
        patience=400,
        lr=0.1,
        hidden_size=1,
        num_layers=1,
        dropout=0.1,
        decay_lr=0.9,
        weight_decay=0.01,
        alias_experiment=ALIAS,
    )

    acc9_test, mse_test, acc1_test = [], [], []

    for split in range(5):
        ids_train = list(splits[str(split)]["train"].keys())
        ids_val = list(splits[str(split)]["val"].keys())
        ids_test = list(splits[str(split)]["test"].keys())
        y_train = pd.Series(splits[str(split)]["train"].values())
        y_val = pd.Series(splits[str(split)]["val"].values())
        y_test = pd.Series(splits[str(split)]["test"].values())

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
        clf.load_model(f"checkpoints/{ALIAS}/split_{split}.ckpt")

        pred_test = clf.predict(scaler.transform(X_test))

        acc9 = balanced_accuracy_score(y_true=y_test, y_pred=pred_test)
        mse = get_mse_macro(y_true=y_test, y_pred=pred_test)
        acc1 = get_acc1_macro(y_true=y_test, y_pred=pred_test)

        acc9_test.append(acc9)
        mse_test.append(mse)
        acc1_test.append(acc1)
        print(f"split {split}: acc9={acc9:.4f} mse={mse:.4f} acc1={acc1:.4f}")

    print("\n=== CIPI reproduction, PRISTINE original code (basic+LZ, 5-fold) ===")
    print(f"Acc-9: {mean(acc9_test) * 100:.1f} (+/-{stdev(acc9_test) * 100:.1f})   [paper: 41.4 (+/-3.1)]")
    print(f"MSE:   {mean(mse_test):.1f} (+/-{stdev(mse_test):.1f})   [paper: 1.7 (+/-0.5)]")
    print(f"Acc-1: {mean(acc1_test) * 100:.1f} (+/-{stdev(acc1_test) * 100:.1f})")


if __name__ == "__main__":
    main()
