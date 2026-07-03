"""
Phase 3: final Guitar RubricNet training/evaluation.

Uses the corrected 13-descriptor feature list (guitar/prepare_splits.py) and
the fixed 5-fold splits (guitar/guitar_splits.json). Loads tuned
hyperparameters from guitar/best_hyperparams_guitar_all.json if present
(produced by guitar/optuna_guitar_tuning.py), otherwise falls back to
untuned defaults.
"""
import json
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler

from guitar.baselines import Args, get_fold_xy, load_data, score, summarize
from guitar.prepare_splits import ALL_FEATURES, NUM_CLASSES
from rubricnet.rubricnet import RubricnetSklearn

DEFAULT_HYPERPARAMS = dict(
    lr=0.005,
    batch_size=16,
    hidden_size=1,
    num_layers=1,
    dropout=0.05,
    decay_lr=0.5,
    weight_decay=1e-4,
    patience=20,
)

ALIAS_EXPERIMENT = "guitar_rubricnet_final"
BEST_HYPERPARAMS_PATH = "guitar/best_hyperparams_guitar_all.json"


def load_hyperparams():
    if os.path.exists(BEST_HYPERPARAMS_PATH):
        with open(BEST_HYPERPARAMS_PATH) as f:
            tuned = json.load(f)["params"]
        hyperparams = dict(DEFAULT_HYPERPARAMS)
        hyperparams.update(tuned)
        print(f"Loaded tuned hyperparameters from {BEST_HYPERPARAMS_PATH}: {tuned}")
        return hyperparams
    print(f"No tuned hyperparameters found at {BEST_HYPERPARAMS_PATH}, using defaults.")
    return dict(DEFAULT_HYPERPARAMS)


def main():
    features, splits = load_data()
    hyperparams = load_hyperparams()

    fold_scores = []
    for split_idx in range(5):
        X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
        X_val, y_val = get_fold_xy(features, splits, split_idx, "val")
        X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

        scaler = StandardScaler().fit(X_train)
        args = Args(alias_experiment=ALIAS_EXPERIMENT, **hyperparams)
        clf = RubricnetSklearn(
            input_dim=len(ALL_FEATURES), num_classes=NUM_CLASSES, split=split_idx, args=args, logging=False
        )
        clf.fit(
            scaler.transform(X_train), y_train,
            scaler.transform(X_val), y_val,
            scaler.transform(X_test), y_test,
        )
        clf.load_model(f"checkpoints/{ALIAS_EXPERIMENT}/split_{split_idx}.ckpt")

        y_pred = clf.predict(scaler.transform(X_test)).cpu().numpy()
        fold_scores.append(score(y_test, y_pred))
        print(f"  split {split_idx}: acc={fold_scores[-1]['accuracy']:.4f} MAE={fold_scores[-1]['mae']:.4f}")

    metrics = summarize("Guitar RubricNet", fold_scores)

    out_path = "guitar/rubricnet_results.json"
    with open(out_path, "w") as f:
        json.dump({"hyperparams": hyperparams, "metrics": metrics}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
