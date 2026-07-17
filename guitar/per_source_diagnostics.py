"""Per-piece predictions broken out by data source (pdf / dada_gp / gaps), for
diagnosing whether errors are concentrated in a specific data source (extraction/
provenance issue) or spread uniformly (model/code issue).

Reuses the exact frozen-split protocol from guitar/baselines.py,
guitar/train_guitar_rubricnet.py and guitar/run_fuzzy_baselines.py:
- RubricNet V3: loads the existing checkpoints/guitar_rubricnet_final_v3_seed_{0,1,2}
  models (no retraining) and predicts on each fold's test split.
- Fuzzy Complete Search / Fuzzy Pattern Tree: refit per fold (deterministic,
  cheap), identical hyperparameter selection as run_fuzzy_baselines.py defaults.

Writes guitar/per_source_predictions.csv with one row per (model, piece,
seed-or-null) prediction, tagged with the piece's source.

--v5 switches everything to the V5 dataset/model: guitar_descriptors_v5.csv,
guitar_splits_v5.json, the guitar_rubricnet_final_v5_seed_{0,1,2} checkpoints
and best_hyperparams_guitar_all_v5.json, writing per_source_predictions_v5.csv.
"""
import argparse
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from guitar.baselines import Args, get_fold_xy
from guitar.fuzzy_rules import Fuzzifier
from guitar.prepare_splits import ALL_FEATURES_V3, NUM_CLASSES, make_piece_id
from guitar.run_fuzzy_baselines import run_complete_search, run_fpt
from guitar.train_guitar_rubricnet import DEFAULT_HYPERPARAMS
from rubricnet.rubricnet import RubricnetSklearn

N_SPLITS = 5
SEEDS = [0, 1, 2]


def load_hyperparams(path):
    with open(path) as f:
        tuned = json.load(f)["params"]
    hyperparams = dict(DEFAULT_HYPERPARAMS)
    hyperparams.update(tuned)
    return hyperparams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5", action="store_true",
                        help="Use the V5 dataset/checkpoints (76 pdf/no-rhythm pieces dropped)")
    args = parser.parse_args()

    if args.v5:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        rubricnet_alias = "guitar_rubricnet_final_v5_seed_{seed}"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/per_source_predictions_v5.csv"
    else:
        csv_path = "features/guitar_descriptors_v3.csv"
        splits_path = "guitar/guitar_splits.json"
        rubricnet_alias = "guitar_rubricnet_final_v3_seed_{seed}"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v3.json"
        out_path = "guitar/per_source_predictions.csv"

    df = pd.read_csv(csv_path)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    source_map = df.set_index("piece_id")["source"].to_dict()
    features = df.set_index("piece_id")[ALL_FEATURES_V3]

    with open(splits_path) as f:
        splits = json.load(f)

    hyperparams = load_hyperparams(best_hyperparams_path)

    rows = []

    for split_idx in range(N_SPLITS):
        print(f"=== Fold {split_idx} ===")
        X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
        X_val, y_val = get_fold_xy(features, splits, split_idx, "val")
        X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

        medians = X_train.median().fillna(0.0)
        X_train_f = X_train.fillna(medians)
        X_val_f = X_val.fillna(medians)
        X_test_f = X_test.fillna(medians)

        # ---------------- RubricNet V3 (load existing checkpoints) ----------------
        scaler = StandardScaler().fit(X_train_f)
        X_test_scaled = scaler.transform(X_test_f)

        for seed in SEEDS:
            alias = rubricnet_alias.format(seed=seed)
            args_cls = Args(alias_experiment=alias, **hyperparams)
            clf = RubricnetSklearn(
                input_dim=len(ALL_FEATURES_V3), num_classes=NUM_CLASSES,
                split=split_idx, args=args_cls, logging=False,
            )
            clf.load_model(f"checkpoints/{alias}/split_{split_idx}.ckpt")
            y_pred = clf.predict(X_test_scaled).cpu().numpy()
            y_pred = np.clip(y_pred, 0, NUM_CLASSES - 1)

            for pid, yt, yp in zip(X_test.index, y_test.values, y_pred):
                rows.append(dict(
                    model="RubricNet", fold=split_idx, seed=seed,
                    piece_id=pid, source=source_map[pid],
                    y_true=int(yt), y_pred=int(yp),
                ))
        print(f"  RubricNet: {len(SEEDS)} seeds x {len(X_test)} test pieces done")

        # ---------------- Fuzzy classifiers (refit per fold, deterministic) ----------------
        X_trainval_f = pd.concat([X_train_f, X_val_f])
        y_trainval = np.concatenate([y_train.values, y_val.values])

        fz = Fuzzifier(method="cdf").fit(X_train_f.values)
        M_train = fz.transform(X_train_f.values)
        M_val = fz.transform(X_val_f.values)

        fz_trainval = Fuzzifier(method="cdf").fit(X_trainval_f.values)
        M_trainval = fz_trainval.transform(X_trainval_f.values)
        M_test = fz_trainval.transform(X_test_f.values)

        y_pred_cs, best_m, _ = run_complete_search(
            M_train, y_train.values, M_val, y_val.values,
            M_trainval, y_trainval, M_test, ALL_FEATURES_V3,
        )
        for pid, yt, yp in zip(X_test.index, y_test.values, y_pred_cs):
            rows.append(dict(
                model="Fuzzy Complete Search", fold=split_idx, seed=None,
                piece_id=pid, source=source_map[pid],
                y_true=int(yt), y_pred=int(yp),
            ))
        print(f"  Fuzzy Complete Search: m={best_m}")

        y_pred_fpt, best_dmax, _, _ = run_fpt(
            M_train, y_train.values, M_val, y_val.values,
            M_trainval, y_trainval, M_test, ALL_FEATURES_V3,
            gamma=0.05, use_negation=True, balanced_rmse=True,
        )
        for pid, yt, yp in zip(X_test.index, y_test.values, y_pred_fpt):
            rows.append(dict(
                model="Fuzzy Pattern Tree", fold=split_idx, seed=None,
                piece_id=pid, source=source_map[pid],
                y_true=int(yt), y_pred=int(yp),
            ))
        print(f"  Fuzzy Pattern Tree: d_max={best_dmax}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_df)} rows)")

    out_df["abs_err"] = (out_df["y_true"] - out_df["y_pred"]).abs()
    out_df["correct"] = out_df["y_true"] == out_df["y_pred"]
    print("\n=== Sanity check: pooled accuracy / MAE per model (should match published numbers) ===")
    print(out_df.groupby("model")[["correct", "abs_err"]].mean())
    print("\n=== Accuracy per model x source ===")
    print(out_df.groupby(["model", "source"])["correct"].agg(["mean", "count"]))
    print("\n=== MAE per model x source ===")
    print(out_df.groupby(["model", "source"])["abs_err"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
