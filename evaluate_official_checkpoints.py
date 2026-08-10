"""
Evaluate the ACTUAL original-authors' trained checkpoints (not a retrain) from
/home/aser/programming/rubricnet -- a fresh clone of github.com/PRamoneda/rubricnet
whose checkpoints/rubricnet_cameraready/ contains their real split_{0-4}.ckpt
model weights AND scaler_{0-4}.pkl fitted scalers, committed directly by Pedro
Ramoneda (the paper's first author). Pure inference, no training at all --
this should reproduce the literal Table 3/4 numbers: Acc-9 = 41.4 (+/-3.1),
MSE = 1.7 (+/-0.5).

Uses the pristine rubricnet_original code (cumprod decode) since that's what
the original authors' repo actually runs, not this thesis repo's guitar-adapted
num_classes>8 branch.
"""
import json
import os
import sys
from statistics import mean, stdev

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubricnet_original.rubricnet import RubricnetSklearn

CLONE_ROOT = "/home/aser/programming/rubricnet"
SPLITS_PATH = f"{CLONE_ROOT}/rubricnet/cipi_splits.json"
FEATURES_PATH = f"{CLONE_ROOT}/features/cipi-features-ISMIR24.json"
CKPT_DIR = f"{CLONE_ROOT}/checkpoints/rubricnet_cameraready"
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

    # Values here don't affect inference (only input_dim/num_classes/hidden_size/
    # num_layers determine the loaded architecture's shapes); dropout is inert
    # in eval mode.
    args = Args(
        batch_size=40, patience=400, lr=0.1, hidden_size=1, num_layers=1,
        dropout=0.1, decay_lr=0.9, weight_decay=0.01, alias_experiment="official_ckpt_eval",
    )

    acc9_test, mse_test, acc1_test = [], [], []

    for split in range(5):
        ids_test = list(splits[str(split)]["test"].keys())
        y_test = pd.Series(splits[str(split)]["test"].values())
        X_test = pd.DataFrame({ft: [features[i][ft] for i in ids_test] for ft in BASIC_FEATURES})

        scaler = pd.read_pickle(f"{CKPT_DIR}/scaler_{split}.pkl")

        clf = RubricnetSklearn(
            input_dim=len(BASIC_FEATURES), num_classes=NUM_CLASSES, split=split, args=args, logging=False
        )
        clf.load_model(f"{CKPT_DIR}/split_{split}.ckpt")

        pred_test = clf.predict(scaler.transform(X_test))

        acc9 = balanced_accuracy_score(y_true=y_test, y_pred=pred_test)
        mse = get_mse_macro(y_true=y_test, y_pred=pred_test)
        acc1 = get_acc1_macro(y_true=y_test, y_pred=pred_test)

        acc9_test.append(acc9)
        mse_test.append(mse)
        acc1_test.append(acc1)
        print(f"split {split}: acc9={acc9:.4f} mse={mse:.4f} acc1={acc1:.4f}")

    print("\n=== Official rubricnet_cameraready checkpoints, pure inference ===")
    print(f"Acc-9: {mean(acc9_test) * 100:.1f} (+/-{stdev(acc9_test) * 100:.1f})   [paper: 41.4 (+/-3.1)]")
    print(f"MSE:   {mean(mse_test):.1f} (+/-{stdev(mse_test):.1f})   [paper: 1.7 (+/-0.5)]")
    print(f"Acc-1: {mean(acc1_test) * 100:.1f} (+/-{stdev(acc1_test) * 100:.1f})")


if __name__ == "__main__":
    main()
