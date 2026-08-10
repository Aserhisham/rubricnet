"""
Bayesian hyperparameter search for the CIPI "basic+LZ" RubricNet reproduction,
mirroring the paper's own search space (Sec. 5.1): batch size 16-128, dropout
0.1-0.5, lr decay 0.1-0.9, lr log-uniform 1e-5 to 1e-1. weight_decay/hidden_size
/num_layers are fixed (paper doesn't tune them either).

Each trial trains all 5 folds with a short patience (fast signal, not the
final reported number) and logs mean val Acc-9 / MSE as a 2-objective Optuna
study. After the search, run retrain_best() to redo the winning config with
the paper's full patience=400 for the number that actually gets reported.

Usage:
    python retune_cipi_paper.py search --n-trials 30
    python retune_cipi_paper.py retrain-best
"""
import json
import os
import sys
from statistics import mean, stdev

import optuna
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubricnet.rubricnet import RubricnetSklearn

SPLITS_PATH = "rubricnet/cipi_splits.json"
FEATURES_PATH = "features/cipi-features-ISMIR24.json"
NUM_CLASSES = 9
STUDY_NAME = "cipi_paper_retune"
STORAGE = f"sqlite:///{STUDY_NAME}.db"

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


def load_data():
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    with open(FEATURES_PATH) as f:
        features = {row["id"]: row for row in json.load(f)}
    return splits, features


def run_5fold(args, alias, splits, features):
    acc9_val, mse_val, acc9_test, mse_test = [], [], [], []
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
        clf.load_model(f"checkpoints/{alias}/split_{split}.ckpt")

        pred_val = clf.predict(scaler.transform(X_val))
        pred_test = clf.predict(scaler.transform(X_test))

        acc9_val.append(balanced_accuracy_score(y_true=y_val, y_pred=pred_val))
        mse_val.append(get_mse_macro(y_true=y_val, y_pred=pred_val))
        acc9_test.append(balanced_accuracy_score(y_true=y_test, y_pred=pred_test))
        mse_test.append(get_mse_macro(y_true=y_test, y_pred=pred_test))
    return acc9_val, mse_val, acc9_test, mse_test


def objective(trial, splits, features):
    args = Args(
        batch_size=trial.suggest_int("batch_size", 16, 128),
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        decay_lr=trial.suggest_float("decay_lr", 0.1, 0.9),
        lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        patience=30,
        hidden_size=1,
        num_layers=1,
        weight_decay=0.01,
        alias_experiment=f"{STUDY_NAME}_{trial.number}",
    )
    acc9_val, mse_val, acc9_test, mse_test = run_5fold(args, args.alias_experiment, splits, features)
    trial.set_user_attr("acc9_val_mean", mean(acc9_val))
    trial.set_user_attr("mse_val_mean", mean(mse_val))
    trial.set_user_attr("acc9_test_mean", mean(acc9_test))
    trial.set_user_attr("acc9_test_std", stdev(acc9_test))
    trial.set_user_attr("mse_test_mean", mean(mse_test))
    trial.set_user_attr("mse_test_std", stdev(mse_test))
    print(f"[trial {trial.number}] acc9_val={mean(acc9_val):.4f} mse_val={mean(mse_val):.4f} "
          f"acc9_test={mean(acc9_test):.4f} mse_test={mean(mse_test):.4f} params={trial.params}")
    return mean(acc9_val), mean(mse_val)


def search(n_trials):
    splits, features = load_data()
    study = optuna.create_study(
        study_name=STUDY_NAME, directions=["maximize", "minimize"],
        storage=STORAGE, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=2),
    )
    study.optimize(lambda t: objective(t, splits, features), n_trials=n_trials, n_jobs=1)

    print("\n=== Pareto-optimal trials (val Acc-9, val MSE) ===")
    best_by_acc = max(study.best_trials, key=lambda t: t.values[0])
    for t in study.best_trials:
        print(f"trial {t.number}: acc9_val={t.values[0]:.4f} mse_val={t.values[1]:.4f} "
              f"test_acc9={t.user_attrs['acc9_test_mean']:.4f} params={t.params}")
    print(f"\nPicked (highest val Acc-9): trial {best_by_acc.number} -> {best_by_acc.params}")
    with open("best_hyperparams_cipi_paper.json", "w") as f:
        json.dump(best_by_acc.params, f, indent=2)


def retrain_best():
    with open("best_hyperparams_cipi_paper.json") as f:
        best_params = json.load(f)
    splits, features = load_data()
    args = Args(
        batch_size=best_params["batch_size"],
        dropout=best_params["dropout"],
        decay_lr=best_params["decay_lr"],
        lr=best_params["lr"],
        patience=400,
        hidden_size=1,
        num_layers=1,
        weight_decay=0.01,
        alias_experiment="cipi_paper_reproduction_retuned",
    )
    acc9_val, mse_val, acc9_test, mse_test = run_5fold(args, args.alias_experiment, splits, features)
    print("\n=== Retuned CIPI reproduction, full patience=400 retrain ===")
    print(f"params: {best_params}")
    print(f"Acc-9: {mean(acc9_test) * 100:.1f} (+/-{stdev(acc9_test) * 100:.1f})   [paper: 41.4 (+/-3.1)]")
    print(f"MSE:   {mean(mse_test):.1f} (+/-{stdev(mse_test):.1f})   [paper: 1.7 (+/-0.5)]")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("search", "retrain-best"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "search":
        n = 30
        if "--n-trials" in sys.argv:
            n = int(sys.argv[sys.argv.index("--n-trials") + 1])
        search(n)
    else:
        retrain_best()
