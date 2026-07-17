"""
Phase 3: Optuna multi-objective (accuracy + MSE) hyperparameter search for
guitar RubricNet. Adapted from rubricnet/optuna_bayesian_optimization.py --
the search loop itself is unchanged from the piano pipeline; only data
loading (guitar descriptors CSV / guitar_splits.json) and the FEATURES
branches (guitar descriptor groups instead of CIPI's music21/jSymbolic sets)
differ.

FEATURES also exposes the lh_only/rh_only/global_only/lh_rh subsets so the
same switch can drive the Phase 4 ablation study without further changes.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statistics import mean, stdev

import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import (
    ALL_FEATURES, ALL_FEATURES_V2, ALL_FEATURES_V3, ALL_FEATURES_V3_PRUNED,
    ALL_FEATURES_V3_BASE, ALL_FEATURES_V3_BASE_NEW, FEATURE_GROUPS, NUM_CLASSES, make_piece_id,
)
from rubricnet.rubricnet import RubricnetSklearn
from guitar.train_guitar_rubricnet import map_20_to_8_numpy

N_SPLITS = 5

FEATURE_SETS = {
    "guitar_all": ALL_FEATURES,
    "guitar_all_v2": ALL_FEATURES_V2,
    "guitar_all_v3": ALL_FEATURES_V3,
    "guitar_all_v3_pruned": ALL_FEATURES_V3_PRUNED,
    "guitar_all_v3_smooth": ALL_FEATURES_V3,
    "guitar_all_v3_base": ALL_FEATURES_V3_BASE,
    "guitar_all_v3_base_new": ALL_FEATURES_V3_BASE_NEW,
    "guitar_all_v4": ALL_FEATURES_V3,
    "guitar_all_v5": ALL_FEATURES_V3,
    "lh_only": FEATURE_GROUPS["lh"],
    "rh_only": FEATURE_GROUPS["rh"],
    "global_only": FEATURE_GROUPS["global"],
    "lh_rh": FEATURE_GROUPS["lh"] + FEATURE_GROUPS["rh"],
}

# Set by main() before study.optimize() so objective() can see them.
FEATURES = "guitar_all"
FEATURES_DATA, SPLITS_DATA = None, None
RAW_DIFFICULTY_MAP = None
RAW_LEVELS = False


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_mse_macro(y_true, y_pred):
    mse_each_class = []
    for true_class in set(y_true):
        tt, pp = zip(*[[tt, pp] for tt, pp in zip(y_true, y_pred) if tt == true_class])
        mse_each_class.append(mean_squared_error(y_true=tt, y_pred=pp))
    return mean(mse_each_class)


def objective(trial):
    args = Args(
        batch_size=trial.suggest_int("batch_size", 16, 64),
        patience=20,
        alias_experiment=f"{trial.study.study_name}_{trial.number}",
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        hidden_size=1,
        num_layers=1,
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        decay_lr=trial.suggest_float("decay_lr", 0.3, 0.9),
        lr=trial.suggest_float("lr", 5e-3, 1e-1, log=True),
        label_smoothing_temp=trial.suggest_float("label_smoothing_temp", 0.0, 0.4),
    )

    # Clean old checkpoints for this trial to avoid versioning conflicts
    import shutil
    checkpoint_dir = f"checkpoints/{args.alias_experiment}"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    n_features = FEATURES_DATA.shape[1]
    acc_val, acc_test, mse_val, mse_test = [], [], [], []

    for split in range(N_SPLITS):
        X_train, y_train = get_fold_xy(FEATURES_DATA, SPLITS_DATA, split, "train")
        X_val, y_val = get_fold_xy(FEATURES_DATA, SPLITS_DATA, split, "val")
        X_test, y_test = get_fold_xy(FEATURES_DATA, SPLITS_DATA, split, "test")

        # Train-fold median imputation
        medians = X_train.median().fillna(0.0)
        X_train = X_train.fillna(medians)
        X_val = X_val.fillna(medians)
        X_test = X_test.fillna(medians)

        scaler = StandardScaler().fit(X_train)

        clf = RubricnetSklearn(
            input_dim=n_features,
            num_classes=20 if RAW_LEVELS else NUM_CLASSES,
            split=split,
            args=args,
            logging=False
        )

        if RAW_LEVELS:
            y_train_fit = pd.Series([RAW_DIFFICULTY_MAP[i] for i in X_train.index], index=X_train.index)
            y_val_fit = pd.Series([RAW_DIFFICULTY_MAP[i] for i in X_val.index], index=X_val.index)
            y_test_fit = pd.Series([RAW_DIFFICULTY_MAP[i] for i in X_test.index], index=X_test.index)
        else:
            y_train_fit, y_val_fit, y_test_fit = y_train, y_val, y_test

        clf.fit(
            scaler.transform(X_train), y_train_fit,
            scaler.transform(X_val), y_val_fit,
            scaler.transform(X_test), y_test_fit,
        )
        clf.load_model(f"checkpoints/{args.alias_experiment}/split_{split}.ckpt")

        pred_val = clf.predict(scaler.transform(X_val)).cpu().numpy()
        pred_test = clf.predict(scaler.transform(X_test)).cpu().numpy()

        if RAW_LEVELS:
            pred_val = map_20_to_8_numpy(pred_val)
            pred_test = map_20_to_8_numpy(pred_test)

        acc_val.append(balanced_accuracy_score(y_val, pred_val))
        acc_test.append(balanced_accuracy_score(y_test, pred_test))
        mse_val.append(get_mse_macro(y_val, pred_val))
        mse_test.append(get_mse_macro(y_test, pred_test))

    # Clean up checkpoint directory to save disk space
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    trial.set_user_attr("acc_test", mean(acc_test))
    trial.set_user_attr("acc_test_std", stdev(acc_test))
    trial.set_user_attr("mse_test", mean(mse_test))
    trial.set_user_attr("mse_test_std", stdev(mse_test))
    return mean(acc_val), mean(mse_val)


def main():
    global FEATURES, FEATURES_DATA, SPLITS_DATA, RAW_DIFFICULTY_MAP, RAW_LEVELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="guitar_all", choices=list(FEATURE_SETS))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--v4", action="store_true", help="Use V4 features")
    parser.add_argument("--raw-levels", action="store_true", help="Train on raw 1-20 difficulty levels")
    cli_args = parser.parse_args()

    FEATURES = cli_args.features
    RAW_LEVELS = cli_args.raw_levels
    study_name = cli_args.study_name or f"guitar_rubricnet_{FEATURES}"
    if RAW_LEVELS:
        study_name = f"{study_name}_raw"
    
    # Load correct CSV/splits path dynamically
    splits_path = "guitar/guitar_splits.json"
    if "_v5" in FEATURES:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
    elif cli_args.v4 or "v4" in FEATURES:
        csv_path = "features/guitar_descriptors_v4.csv"
    elif "v3_base_new" in FEATURES:
        csv_path = "features/guitar_descriptors_v3_base.csv"
    elif "_v3" in FEATURES:
        csv_path = "features/guitar_descriptors_v3.csv"
    elif "_v2" in FEATURES:
        csv_path = "features/guitar_descriptors_v2.csv"
    else:
        csv_path = "features/guitar_descriptors.csv"

    FEATURES_DATA, SPLITS_DATA = load_data(csv_path=csv_path, splits_path=splits_path, columns=FEATURE_SETS[FEATURES])

    if RAW_LEVELS:
        df_raw = pd.read_csv(csv_path)
        df_raw["piece_id"] = df_raw.apply(make_piece_id, axis=1)
        RAW_DIFFICULTY_MAP = {row["piece_id"]: int(row["Difficulty"]) - 1 for _, row in df_raw.iterrows()}

    sqlite_url = f"sqlite:///guitar/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "minimize"],
        storage=sqlite_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=2),
    )
    study.optimize(objective, n_trials=cli_args.n_trials, n_jobs=1)

    print("\nPareto-optimal trials (val accuracy, val MSE):")
    for t in study.best_trials:
        print(f"  trial {t.number}: acc_val={t.values[0]:.4f} mse_val={t.values[1]:.4f} params={t.params}")

    best = max(study.best_trials, key=lambda t: t.values[0])
    print("\nBest trial by validation accuracy:")
    print(f"  acc_val={best.values[0]:.4f} mse_val={best.values[1]:.4f}")
    print(f"  test: acc={best.user_attrs['acc_test']:.4f} mse={best.user_attrs['mse_test']:.4f}")
    print(f"  params: {best.params}")

    out_path = f"guitar/best_hyperparams_{FEATURES}"
    if RAW_LEVELS:
        out_path = f"{out_path}_raw"
    out_path = f"{out_path}.json"
    
    with open(out_path, "w") as f:
        json.dump({"features": FEATURES, "params": best.params, "trial_number": best.number}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
