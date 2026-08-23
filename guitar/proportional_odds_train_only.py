"""Refit the proportional-odds baseline on the training fold only.

Why
---
`guitar/tuned_baselines.py` pools train+val (512 pieces per fold) as its selection
pool and refits the winning configuration on that pool. RubricNet trains on the
460-piece training fold alone, holding the 52-piece validation fold back for early
stopping. The proportional-odds model therefore sees ~11% more data than the model
it is compared against in Section "The Proportional-Odds Comparison" -- a fifth
difference between the two fits, alongside the shape function, the training
objective, the ordinal head and the decoding rule.

This script removes that difference: the identical search space, budget, inner-CV
protocol, folds and selection criteria, but with the pool restricted to `train`.
If the model's lead over RubricNet survives, the confound is not what produces it.

Usage
-----
    python -m guitar.proportional_odds_train_only
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from guitar.tuned_baselines import (
    FEATURE_SETS,
    INNER_FOLDS,
    N_SPLITS,
    SEARCH_SPACES,
    SEED,
    _with_macro,
    get_fold_xy,
    load_data,
    resolve_scoring,
    summarize,
    _jsonable,
)

OUT_PATH = "guitar/proportional_odds_train_only.json"
CRITERIA = ("balanced_accuracy", "accuracy", "macro_mae")
N_ITER = 60


def train_only_matrices(features, splits, split_idx):
    """Same as tuned_baselines.fold_matrices but the pool is `train` alone."""
    X_pool, y_pool = get_fold_xy(features, splits, split_idx, "train")
    X_test, y_test = get_fold_xy(features, splits, split_idx, "test")
    medians = X_pool.median().fillna(0.0)
    return X_pool.fillna(medians), y_pool, X_test.fillna(medians), y_test


def run(features, splits, scoring):
    estimator, space = SEARCH_SPACES["proportional_odds"]
    fold_metrics, chosen, predictions = [], [], []
    for split_idx in range(N_SPLITS):
        X_pool, y_pool, X_test, y_test = train_only_matrices(features, splits, split_idx)
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=space,
            n_iter=N_ITER,
            scoring=resolve_scoring(scoring),
            cv=inner,
            random_state=SEED + split_idx,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_pool, y_pool)
        chosen.append({k: _jsonable(v) for k, v in search.best_params_.items()})
        y_true = y_test.to_numpy()
        y_pred = search.best_estimator_.predict(X_test)
        m = _with_macro(y_true, y_pred)
        fold_metrics.append(m)
        predictions.append({"fold": split_idx, "seed": SEED,
                            "y_true": [int(v) for v in y_true],
                            "y_pred": [int(v) for v in y_pred]})
        print(f"  split {split_idx}: n_train={len(y_pool)} acc={m['accuracy']:.4f} "
              f"bacc={m['balanced_accuracy']:.4f} MAE={m['mae']:.4f} tau={m['kendall_tau']:.4f}")
    out = summarize(f"proportional_odds_train_only[{scoring}]", fold_metrics)
    out["chosen"] = chosen
    out["predictions"] = predictions
    return out


def main():
    columns, csv_path, splits_path = FEATURE_SETS["v5-pruned-collinear2"]
    features, splits = load_data(csv_path, splits_path, columns)
    results = {"protocol": {
        "feature_set": "v5-pruned-collinear2",
        "n_descriptors": len(columns),
        "outer_folds": N_SPLITS,
        "inner_folds": INNER_FOLDS,
        "n_iter": N_ITER,
        "note": ("selection pool restricted to the 460-piece training fold, matching the "
                 "data RubricNet sees; the 52-piece validation fold is discarded rather "
                 "than pooled, so the comparison is on equal training data."),
    }}
    for scoring in CRITERIA:
        print("\n" + "=" * 78)
        print(f"PROPORTIONAL ODDS, TRAIN-ONLY POOL  (selection criterion: {scoring})")
        print("=" * 78)
        results[f"proportional_odds_train_only[{scoring}]"] = run(features, splits, scoring)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
