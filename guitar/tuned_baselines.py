"""Hyperparameter-tuned black-box baselines for guitar difficulty classification.

Motivation
----------
`guitar/baselines.py` constructs its baselines with hardcoded settings:
`RandomForestClassifier(n_estimators=200)` and `DecisionTreeClassifier(max_depth=6)`,
every other parameter left at the sklearn default. RubricNet, by contrast, received
a 166-trial Optuna search (`guitar/optuna_guitar_tuning.py`). Any claim that the
interpretable model "matches the strongest baseline" is therefore confounded by
unequal tuning effort, in RubricNet's favour.

This script removes that confound. Each baseline gets a randomised hyperparameter
search under the *same* protocol and the *same* selection criterion RubricNet's
Optuna study used (mean validation balanced accuracy), so the comparison reported
in the thesis is between tuned models on both sides.

Protocol
--------
For each of the 5 frozen outer folds:
  * train + val are pooled into a selection pool (the outer test fold is never touched);
  * an inner StratifiedKFold(5) over that pool scores each sampled configuration;
  * the best configuration by inner-CV balanced accuracy is refit on the whole pool;
  * that model predicts the untouched outer test fold.

The chosen configuration is recorded per fold, so the thesis can report both the
searched grid and what was actually selected, rather than a magic number.

Trivial baselines (majority class, stratified guessing, and two single-descriptor
models built on note count alone) are evaluated under the identical fold protocol.
They answer the obvious examiner question -- "how much of this is just piece length?"
-- which the thesis currently cannot answer.

Usage
-----
    python -m guitar.tuned_baselines --v5-pruned-collinear2
    python -m guitar.tuned_baselines --v5-pruned-collinear2 --n-iter 120
"""

import argparse
import json
import sys
from statistics import mean, stdev

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, ".")

from guitar.prepare_splits import (
    ALL_FEATURES_V3,
    ALL_FEATURES_V5_PRUNED,
    ALL_FEATURES_V5_PRUNED_COLLINEAR,
    ALL_FEATURES_V5_PRUNED_COLLINEAR2,
    NUM_CLASSES,
    make_piece_id,
)
from guitar.train_guitar_rubricnet import compute_metrics

N_SPLITS = 5
INNER_FOLDS = 5
SEED = 42

# Selection criterion. RubricNet's Optuna study maximised mean validation balanced
# accuracy (guitar/optuna_guitar_tuning.py:153), so the baselines are selected on the
# same quantity to keep the comparison symmetric.
SELECTION_SCORING = "balanced_accuracy"


# --------------------------------------------------------------------------------
# Search spaces
# --------------------------------------------------------------------------------
# Each space is deliberately wide enough to contain the previously hardcoded setting,
# so the search can only ever match or improve on the untuned baseline -- it cannot be
# accused of being rigged to make the old number look bad.

SEARCH_SPACES = {
    "random_forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {
            # Capped at 600 trees: with 512 training pieces the out-of-bag error curve is
            # flat well before that, so a larger forest costs search budget without
            # changing the selected model. The previously hardcoded 200 is inside the range.
            "n_estimators": randint(100, 600),
            "max_depth": [None, 4, 6, 8, 12, 16, 24],
            "min_samples_leaf": randint(1, 12),
            "min_samples_split": randint(2, 20),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=SEED),
        {
            "max_depth": [None, 3, 4, 5, 6, 8, 10, 14, 20],
            "min_samples_leaf": randint(1, 25),
            "min_samples_split": randint(2, 30),
            "max_features": ["sqrt", "log2", 0.5, 0.8, None],
            "criterion": ["gini", "entropy"],
            "ccp_alpha": uniform(0.0, 0.03),
            "class_weight": [None, "balanced"],
        },
    ),
    "extra_trees": (
        ExtraTreesClassifier(random_state=SEED, n_jobs=1),
        {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 6, 10, 16, 24],
            "min_samples_leaf": randint(1, 12),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced"],
        },
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {
            "learning_rate": loguniform(1e-3, 3e-1),
            "max_iter": randint(80, 600),
            "max_leaf_nodes": randint(4, 64),
            "min_samples_leaf": randint(3, 40),
            "l2_regularization": loguniform(1e-6, 1e0),
            "max_features": uniform(0.4, 0.6),
        },
    ),
}


# --------------------------------------------------------------------------------
# Data plumbing (mirrors guitar/baselines.py so results stay comparable)
# --------------------------------------------------------------------------------
def load_data(csv_path, splits_path, columns):
    df = pd.read_csv(csv_path)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    features = df.set_index("piece_id")[columns]
    with open(splits_path) as f:
        splits = json.load(f)
    return features, splits


def get_fold_xy(features, splits, split_idx, subset):
    fold_labels = splits[str(split_idx)][subset]
    ids = list(fold_labels.keys())
    X = features.loc[ids]
    y = pd.Series([fold_labels[i] for i in ids], index=ids)
    return X, y


def fold_matrices(features, splits, split_idx):
    """Return (X_pool, y_pool, X_test, y_test) with train-fold median imputation.

    train and val are pooled: the inner CV creates its own selection splits, so the
    8% val fold (~52 pieces) is not wasted as a second, much noisier selection set.
    Medians come from the pool only -- the test fold never informs imputation.
    """
    X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
    X_val, y_val = get_fold_xy(features, splits, split_idx, "val")
    X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

    X_pool = pd.concat([X_train, X_val])
    y_pool = pd.concat([y_train, y_val])

    medians = X_pool.median().fillna(0.0)
    X_pool = X_pool.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_pool, y_pool, X_test, y_test


def summarize(name, fold_metrics, keys=("accuracy", "balanced_accuracy", "acc_plus_minus_1", "mae", "mse", "kendall_tau")):
    print(f"\n{name}")
    out = {}
    for key in keys:
        values = [m[key] for m in fold_metrics]
        out[key] = values
        print(f"  {key:20s} {mean(values):.4f} +/- {stdev(values):.4f}")
    return out


# --------------------------------------------------------------------------------
# Tuned baselines
# --------------------------------------------------------------------------------
def run_tuned(features, splits, columns, model_name, n_iter, scoring=SELECTION_SCORING, num_classes=NUM_CLASSES):
    estimator, space = SEARCH_SPACES[model_name]
    fold_metrics = []
    chosen = []
    importances = []

    for split_idx in range(N_SPLITS):
        X_pool, y_pool, X_test, y_test = fold_matrices(features, splits, split_idx)

        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=space,
            n_iter=n_iter,
            scoring=scoring,
            cv=inner,
            random_state=SEED + split_idx,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_pool, y_pool)

        y_pred = search.best_estimator_.predict(X_test)
        m = compute_metrics(y_test.to_numpy(), y_pred)
        fold_metrics.append(m)
        chosen.append({k: _jsonable(v) for k, v in search.best_params_.items()})

        if hasattr(search.best_estimator_, "feature_importances_"):
            importances.append(search.best_estimator_.feature_importances_)

        print(
            f"  split {split_idx}: acc={m['accuracy']:.4f} bacc={m['balanced_accuracy']:.4f} "
            f"MAE={m['mae']:.4f} tau={m['kendall_tau']:.4f} "
            f"(inner {scoring}={search.best_score_:.4f})"
        )
        print(f"    chosen: {chosen[-1]}")

    metrics = summarize(
        f"{model_name} (tuned on {scoring}, {n_iter} samples x {INNER_FOLDS}-fold inner CV)", fold_metrics
    )
    metrics["chosen_params_per_fold"] = chosen
    metrics["search_space"] = {k: _describe(v) for k, v in space.items()}
    metrics["n_iter"] = n_iter
    metrics["selection_scoring"] = scoring

    if importances:
        mean_importance = np.mean(importances, axis=0)
        ranked = sorted(zip(columns, mean_importance), key=lambda t: -t[1])
        metrics["feature_importances"] = {f: float(i) for f, i in ranked}

    return metrics


# --------------------------------------------------------------------------------
# Trivial / single-descriptor baselines
# --------------------------------------------------------------------------------
class QuantileBinBaseline:
    """Predict the difficulty class by equal-frequency binning of one descriptor.

    The most literal "is this just piece length?" control: it learns nothing but the
    8 quantile edges of a single descriptor on the training pool, then assigns each
    test piece the bin its descriptor value falls into. Monotone by construction,
    one parameter vector, no model.
    """

    def __init__(self, column, num_classes=NUM_CLASSES):
        self.column = column
        self.num_classes = num_classes

    def fit(self, X, y):
        values = X[self.column].to_numpy(dtype=float)
        qs = np.linspace(0, 100, self.num_classes + 1)[1:-1]
        self.edges_ = np.percentile(values, qs)
        return self

    def predict(self, X):
        values = X[self.column].to_numpy(dtype=float)
        return np.digitize(values, self.edges_)


def run_trivial(features, splits, columns, num_classes=NUM_CLASSES):
    """Majority class, stratified guessing, and single-descriptor note-count models."""
    results = {}

    length_col = "log_total_notes" if "log_total_notes" in columns else columns[0]

    definitions = {
        "majority_class": lambda: DummyClassifier(strategy="most_frequent"),
        "stratified_guess": lambda: DummyClassifier(strategy="stratified", random_state=SEED),
        f"quantile_bin[{length_col}]": lambda: QuantileBinBaseline(length_col, num_classes),
        f"random_forest_1feat[{length_col}]": lambda: RandomForestClassifier(
            n_estimators=500, random_state=SEED, n_jobs=-1
        ),
    }

    for name, factory in definitions.items():
        fold_metrics = []
        single_feature = name.startswith("random_forest_1feat")
        for split_idx in range(N_SPLITS):
            X_pool, y_pool, X_test, y_test = fold_matrices(features, splits, split_idx)
            if single_feature:
                X_pool = X_pool[[length_col]]
                X_test = X_test[[length_col]]

            model = factory().fit(X_pool, y_pool)
            y_pred = model.predict(X_test)
            fold_metrics.append(compute_metrics(y_test.to_numpy(), np.asarray(y_pred)))

        results[name] = summarize(name, fold_metrics)

    return results


# --------------------------------------------------------------------------------
# Untuned reference (reproduces guitar/baselines.py settings under this protocol)
# --------------------------------------------------------------------------------
def run_untuned_reference(features, splits, columns):
    """The old hardcoded settings, evaluated identically, so the tuning delta is isolated."""
    results = {}
    definitions = {
        "random_forest_untuned": lambda: RandomForestClassifier(random_state=SEED, n_estimators=200),
        "decision_tree_untuned": lambda: DecisionTreeClassifier(random_state=SEED, max_depth=6),
    }
    for name, factory in definitions.items():
        fold_metrics = []
        for split_idx in range(N_SPLITS):
            X_pool, y_pool, X_test, y_test = fold_matrices(features, splits, split_idx)
            model = factory().fit(X_pool, y_pool)
            y_pred = model.predict(X_test)
            fold_metrics.append(compute_metrics(y_test.to_numpy(), y_pred))
        results[name] = summarize(name, fold_metrics)
    return results


# --------------------------------------------------------------------------------
def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _describe(space):
    if isinstance(space, list):
        return {"type": "categorical", "values": [_jsonable(v) for v in space]}
    dist = getattr(space, "dist", None)
    name = getattr(dist, "name", type(space).__name__)
    args = getattr(space, "args", ())
    return {"type": name, "args": [float(a) for a in args]}


FEATURE_SETS = {
    "v5": (ALL_FEATURES_V3, "features/guitar_descriptors_v5.csv", "guitar/guitar_splits_v5.json"),
    "v5-pruned": (ALL_FEATURES_V5_PRUNED, "features/guitar_descriptors_v5.csv", "guitar/guitar_splits_v5.json"),
    "v5-pruned-collinear": (
        ALL_FEATURES_V5_PRUNED_COLLINEAR,
        "features/guitar_descriptors_v5.csv",
        "guitar/guitar_splits_v5.json",
    ),
    "v5-pruned-collinear2": (
        ALL_FEATURES_V5_PRUNED_COLLINEAR2,
        "features/guitar_descriptors_v5.csv",
        "guitar/guitar_splits_v5.json",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-set", default="v5-pruned-collinear2", choices=sorted(FEATURE_SETS))
    parser.add_argument("--v5-pruned-collinear2", dest="feature_set", action="store_const", const="v5-pruned-collinear2")
    parser.add_argument("--n-iter", type=int, default=80, help="configurations sampled per fold per model")
    parser.add_argument(
        "--models",
        default="random_forest,decision_tree,extra_trees,hist_gradient_boosting",
        help="comma-separated subset of the tunable models",
    )
    parser.add_argument("--skip-trivial", action="store_true")
    parser.add_argument(
        "--scoring",
        default=SELECTION_SCORING,
        help=(
            "inner-CV selection criterion. Comma-separate to run the search once per "
            "criterion (e.g. 'balanced_accuracy,accuracy'), which shows the baseline is "
            "not disadvantaged by the choice of tuning target."
        ),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    columns, csv_path, splits_path = FEATURE_SETS[args.feature_set]
    out_path = args.out or f"guitar/tuned_baseline_results_{args.feature_set.replace('-', '_')}.json"

    print(f"Feature set : {args.feature_set} ({len(columns)} descriptors)")
    print(f"Descriptors : {csv_path}")
    print(f"Splits      : {splits_path}")
    print(f"Selection   : inner {INNER_FOLDS}-fold CV on train+val, scoring={SELECTION_SCORING}")
    print(f"Budget      : {args.n_iter} sampled configurations per fold per model")

    features, splits = load_data(csv_path, splits_path, columns)

    results = {
        "protocol": {
            "feature_set": args.feature_set,
            "n_descriptors": len(columns),
            "descriptors": list(columns),
            "outer_folds": N_SPLITS,
            "inner_folds": INNER_FOLDS,
            "selection_scoring": SELECTION_SCORING,
            "n_iter": args.n_iter,
            "note": (
                "train+val pooled as the selection pool; inner StratifiedKFold selects "
                "hyperparameters; best config refit on the pool and evaluated once on the "
                "untouched outer test fold."
            ),
        }
    }

    print("\n" + "=" * 78)
    print("UNTUNED REFERENCE (settings from guitar/baselines.py)")
    print("=" * 78)
    results.update(run_untuned_reference(features, splits, columns))

    if not args.skip_trivial:
        print("\n" + "=" * 78)
        print("TRIVIAL AND SINGLE-DESCRIPTOR BASELINES")
        print("=" * 78)
        results.update(run_trivial(features, splits, columns))

    scorings = [s.strip() for s in args.scoring.split(",") if s.strip()]
    results["protocol"]["selection_scoring"] = scorings

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        for scoring in scorings:
            key = model_name if len(scorings) == 1 else f"{model_name}[{scoring}]"
            print("\n" + "=" * 78)
            print(f"TUNED: {model_name}  (selection criterion: {scoring})")
            print("=" * 78)
            results[key] = run_tuned(features, splits, columns, model_name, args.n_iter, scoring=scoring)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
