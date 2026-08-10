"""
Confusion matrix figure for the adopted V5-pruned-collinear2 (25-feature)
RubricNet model, replacing the old V3 confusion_matrix.pdf in the thesis.

Pools seed-0 test predictions across all 5 folds (same protocol as the
existing V3 figure: one seed, pooled over the five test folds, row-normalized,
cell counts annotated) using the already-trained
checkpoints/guitar_rubricnet_final_v5_pruned_collinear2_seed_0/split_{0-4}.ckpt
checkpoints -- pure inference, no retraining. Scaler/median-imputation refit
per fold from the frozen train split, same as guitar/backfill_rae_v5.py.

Output: AIM-thesis/figures/confusion_matrix_v5.pdf
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("WANDB_MODE", "disabled")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR2, NUM_CLASSES
from guitar.train_guitar_rubricnet import DEFAULT_HYPERPARAMS
from rubricnet.rubricnet import RubricnetSklearn

CSV_PATH = "features/guitar_descriptors_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
ALIAS = "guitar_rubricnet_final_v5_pruned_collinear2_seed_0"
BEST_HYPERPARAMS_PATH = "guitar/best_hyperparams_guitar_all_v5.json"
CKPT_DIR = "checkpoints"
OUT_PATH = "AIM-thesis/figures/confusion_matrix_v5.pdf"


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
    features, splits = load_data(csv_path=CSV_PATH, splits_path=SPLITS_PATH, columns=ALL_FEATURES_V5_PRUNED_COLLINEAR2)
    hyperparams = load_hyperparams(BEST_HYPERPARAMS_PATH)

    all_true, all_pred = [], []
    for split_idx in range(5):
        X_train, _ = get_fold_xy(features, splits, split_idx, "train")
        X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

        medians = X_train.median().fillna(0.0)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        scaler = StandardScaler().fit(X_train)
        args = Args(alias_experiment=ALIAS, **hyperparams)

        clf = RubricnetSklearn(
            input_dim=len(ALL_FEATURES_V5_PRUNED_COLLINEAR2), num_classes=NUM_CLASSES,
            split=split_idx, args=args, logging=False,
        )
        clf.load_model(f"{CKPT_DIR}/{ALIAS}/split_{split_idx}.ckpt")

        y_pred = clf.predict(scaler.transform(X_test)).cpu().numpy()
        y_pred = np.clip(y_pred, 0, NUM_CLASSES - 1)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        acc = float((np.array(y_pred) == np.array(y_test)).mean())
        print(f"  split {split_idx}: acc={acc:.4f} n={len(y_test)}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_acc = float((all_true == all_pred).mean())
    within_1 = float((np.abs(all_true - all_pred) <= 1).mean())
    print(f"\nPooled: acc={overall_acc:.4f}  acc+/-1={within_1:.4f}  n={len(all_true)}")

    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("RubricNet V5-pruned-collinear2 (25 feat)\nConfusion matrix, seed 0, pooled over 5 test folds")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            count = cm[i, j]
            frac = cm_norm[i, j]
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    plt.close()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
