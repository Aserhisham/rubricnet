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
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# Macro-averaged error scorers. RubricNet's Optuna study was multi-objective over
# (validation balanced accuracy, validation macro-MSE), while the baselines above are
# selected on balanced accuracy alone -- an asymmetry that favours RubricNet on exactly
# the macro-averaged measures its headline claim rests on. Registering these lets the
# baselines be re-tuned directly for those measures, so the claim can be tested against
# a forest optimised for the metric it is said to lose on.
# macro_mse matches guitar/optuna_guitar_tuning.py:get_mse_macro exactly.
def _macro_error(y_true, y_pred, power):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(
        np.mean([np.mean(np.abs(y_pred[y_true == c] - y_true[y_true == c]) ** power)
                 for c in np.unique(y_true)])
    )


CUSTOM_SCORERS = {
    "macro_mae": make_scorer(lambda t, p: _macro_error(t, p, 1), greater_is_better=False),
    "macro_mse": make_scorer(lambda t, p: _macro_error(t, p, 2), greater_is_better=False),
}


def resolve_scoring(name):
    """Map a criterion name to a scorer object, passing sklearn's own names through."""
    return CUSTOM_SCORERS.get(name, name)


# --------------------------------------------------------------------------------
# Linear reference models
# --------------------------------------------------------------------------------
# The row labelled "Ordinal Regression" in guitar/baselines.py is NOT a linear model:
# run_rubricnet_default_hparams() instantiates RubricnetSklearn, whose LogisticRegressionOrdinal
# wraps the very same Rubricnet module (rubricnet/rubricnet.py:166 -- the plain
# nn.Linear it once used is commented out on the line above). It is therefore RubricNet
# under untuned hyperparameters, not an independent baseline, and cannot serve as the
# "floor without any nonlinearity" the thesis describes.
#
# OrdinalLogistic below is that missing floor: the Frank & Hall decomposition of an
# ordinal target into K-1 binary "is the class above j?" problems, each solved by plain
# logistic regression. It is linear, interpretable (one coefficient per descriptor per
# threshold), and ordinal -- so pairing it with a plain multinomial logistic regression
# separates two things the thesis currently conflates: what the ordinal encoding buys,
# and what the per-descriptor tanh subnetworks buy on top of it.
class OrdinalLogistic(ClassifierMixin, BaseEstimator):
    """Frank & Hall ordinal decomposition over logistic regression.

    Fits K-1 binary classifiers, the j-th estimating P(y > j), then recovers
    P(y = k) = P(y > k-1) - P(y > k) with the conventions P(y > -1) = 1 and
    P(y > K-1) = 0. Monotonicity of the cumulative probabilities is not enforced, so
    the differences are clipped at zero before the argmax.
    """

    def __init__(self, C=1.0, class_weight=None, max_iter=2000, num_classes=NUM_CLASSES):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.num_classes = num_classes

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.arange(self.num_classes)
        self.models_ = []
        for j in range(self.num_classes - 1):
            target = (y > j).astype(int)
            if len(np.unique(target)) < 2:
                # Degenerate threshold (no training piece on one side of it): fall back
                # to the constant probability rather than failing the whole fit.
                self.models_.append(float(target.mean()))
                continue
            m = LogisticRegression(
                C=self.C, class_weight=self.class_weight, max_iter=self.max_iter
            ).fit(X, target)
            self.models_.append(m)
        return self

    def _cumulative(self, X):
        n = X.shape[0]
        out = np.empty((n, self.num_classes - 1))
        for j, m in enumerate(self.models_):
            out[:, j] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
        return out

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        cum = self._cumulative(X)
        ones = np.ones((X.shape[0], 1))
        zeros = np.zeros((X.shape[0], 1))
        upper = np.hstack([ones, cum])       # P(y > k-1)
        lower = np.hstack([cum, zeros])      # P(y > k)
        probs = np.clip(upper - lower, 0.0, None)
        totals = probs.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return probs / totals

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------------
# The decomposable linear reference.
#
# OrdinalLogistic above is the Frank & Hall decomposition, which fits an independent
# coefficient vector per threshold. That makes it ordinal but NOT decomposable in
# RubricNet's sense: a descriptor has a different effect at every class boundary, so
# no single per-descriptor contribution exists to read off. The thesis used that
# property to argue the additive architecture buys a decomposition the linear model
# cannot offer -- an argument that is an artifact of the decomposition chosen, not of
# linearity.
#
# The proportional-odds (cumulative logit) model is the missing comparison. It shares
# ONE coefficient vector across all thresholds,
#     P(y <= j | x) = sigma(theta_j - x . w),   theta_0 < ... < theta_{K-2},
# so x_i * w_i is exactly descriptor i's contribution to a single latent score, in the
# same sense g_i(x_i) is under RubricNet. Structurally RubricNet IS this model with a
# per-descriptor tanh inserted: S(x) = sum_i tanh(w_i x_i + b_i) against sum_i w_i x_i.
# Running it therefore measures what the saturating shape functions buy over a linear
# additive model that is equally ordinal and equally readable.
class ProportionalOddsLogistic(ClassifierMixin, BaseEstimator):
    """Cumulative-logit ordinal regression with a shared coefficient vector.

    Fitted by direct maximum likelihood (L-BFGS) with an L2 penalty on the
    coefficients scaled as 1/(2C), matching scikit-learn's C convention. Threshold
    ordering is enforced by parametrising theta as (theta_0, softplus(deltas)).
    """

    def __init__(self, C=1.0, class_weight=None, max_iter=500, num_classes=NUM_CLASSES):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.num_classes = num_classes

    @staticmethod
    def _thetas(raw):
        """Map unconstrained parameters to strictly increasing cut points."""
        return np.concatenate([raw[:1], raw[:1] + np.cumsum(np.logaddexp(0.0, raw[1:]))])

    def _cum_probs(self, X, w, thetas):
        """P(y <= j) for j = 0..K-2, shape (n, K-1)."""
        eta = X @ w
        return 1.0 / (1.0 + np.exp(-(thetas[None, :] - eta[:, None])))

    def _class_probs(self, X, w, thetas):
        cum = self._cum_probs(X, w, thetas)
        ones = np.ones((X.shape[0], 1))
        zeros = np.zeros((X.shape[0], 1))
        return np.clip(np.hstack([cum, ones]) - np.hstack([zeros, cum]), 1e-12, None)

    def fit(self, X, y):
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        K = self.num_classes
        n, m = X.shape
        self.classes_ = np.arange(K)

        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=K).astype(float)
            counts[counts == 0] = 1.0
            sample_w = (n / (K * counts))[y]
        else:
            sample_w = np.ones(n)

        def nll(params):
            w = params[:m]
            thetas = self._thetas(params[m:])
            probs = self._class_probs(X, w, thetas)
            ll = -np.sum(sample_w * np.log(probs[np.arange(n), y]))
            return ll + np.dot(w, w) / (2.0 * self.C)

        init = np.concatenate([np.zeros(m), np.linspace(-1.0, 1.0, K - 1)])
        res = minimize(nll, init, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x[:m]
        self.theta_ = self._thetas(res.x[m:])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = self._class_probs(X, self.coef_, self.theta_)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _linear_pipeline(estimator):
    """Linear models need standardized inputs; the trees do not care."""
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


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
    # Linear floor, ordinal encoding (Frank & Hall over logistic regression).
    "ordinal_logistic": (
        _linear_pipeline(OrdinalLogistic()),
        {
            "clf__C": loguniform(1e-3, 1e2),
            "clf__class_weight": [None, "balanced"],
        },
    ),
    # Linear floor, ordinal encoding, ONE coefficient per descriptor (proportional
    # odds). Unlike the Frank & Hall row above, this one is decomposable in exactly
    # RubricNet's sense, so it isolates what the per-descriptor tanh buys.
    "proportional_odds": (
        _linear_pipeline(ProportionalOddsLogistic()),
        {
            "clf__C": loguniform(1e-3, 1e2),
            "clf__class_weight": [None, "balanced"],
        },
    ),
    # Linear floor, nominal encoding. Differs from the row above only in whether the
    # class order is used, which is exactly the contrast the fuzzy comparison claims
    # to isolate but cannot, since the fuzzy methods differ in several ways at once.
    "multinomial_logistic": (
        _linear_pipeline(LogisticRegression(max_iter=2000)),
        {
            "clf__C": loguniform(1e-3, 1e2),
            "clf__class_weight": [None, "balanced"],
        },
    ),
}

# Models whose fit depends on a random seed, and which therefore get the same
# 3-seed treatment RubricNet receives, so that the +- columns mean the same thing
# in every row of the final table.
STOCHASTIC_MODELS = {"random_forest", "decision_tree", "extra_trees", "hist_gradient_boosting"}


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


FULL_METRIC_KEYS = (
    "accuracy", "balanced_accuracy", "acc_plus_minus_1", "mae", "mse",
    "macro_mae", "macro_mse", "kendall_tau", "spearman_rho",
    "cohen_kappa", "linear_weighted_kappa", "quadratic_weighted_kappa",
)


def _with_macro(y_true, y_pred):
    """Every metric the thesis reports anywhere, computed for every model.

    The generation-by-generation table currently leaves acc+-1 and Kendall's tau
    blank for most baselines and reports the macro-averaged and chance-corrected
    measures only for the final configuration. Computing the full suite for every
    row costs nothing and removes the gaps.
    """
    m = compute_metrics(y_true, y_pred)
    m["macro_mae"] = _macro_error(y_true, y_pred, 1)
    m["macro_mse"] = _macro_error(y_true, y_pred, 2)

    rho = spearmanr(y_true, y_pred).statistic
    m["spearman_rho"] = 0.0 if np.isnan(rho) else float(rho)
    labels = list(range(NUM_CLASSES))
    m["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    m["linear_weighted_kappa"] = float(
        cohen_kappa_score(y_true, y_pred, labels=labels, weights="linear")
    )
    m["quadratic_weighted_kappa"] = float(
        cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    )
    return m


def summarize(name, fold_metrics, keys=FULL_METRIC_KEYS):
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
def _seeded(estimator, seed):
    """Return a clone with its random_state set, wherever the estimator exposes one."""
    est = clone(estimator)
    params = est.get_params()
    updates = {k: seed for k in params if k == "random_state" or k.endswith("__random_state")}
    if updates:
        est.set_params(**updates)
    return est


def run_tuned(features, splits, columns, model_name, n_iter, scoring=SELECTION_SCORING,
              num_classes=NUM_CLASSES, seeds=(SEED,)):
    """Tune per fold, then refit the selected configuration under each seed.

    Hyperparameters are selected once per outer fold (as before); the winning
    configuration is then refitted under every seed. This mirrors RubricNet's
    protocol exactly -- one hyperparameter search, then 3 training seeds -- so the
    reported spread covers the same sources of variation on both sides of the
    comparison. Deterministic models are run once regardless of how many seeds are
    requested, since refitting them would only duplicate identical numbers.
    """
    estimator, space = SEARCH_SPACES[model_name]
    effective_seeds = tuple(seeds) if model_name in STOCHASTIC_MODELS else (SEED,)
    fold_metrics = []
    chosen = []
    importances = []
    predictions = []

    for split_idx in range(N_SPLITS):
        X_pool, y_pool, X_test, y_test = fold_matrices(features, splits, split_idx)

        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=space,
            n_iter=n_iter,
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

        for seed in effective_seeds:
            if seed == SEED and len(effective_seeds) == 1:
                model = search.best_estimator_
            else:
                model = _seeded(clone(estimator).set_params(**search.best_params_), seed)
                model.fit(X_pool, y_pool)

            y_pred = model.predict(X_test)
            m = _with_macro(y_true, y_pred)
            fold_metrics.append(m)
            predictions.append({"fold": split_idx, "seed": int(seed),
                                "y_true": [int(v) for v in y_true],
                                "y_pred": [int(v) for v in y_pred]})

            inner_clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
            if hasattr(inner_clf, "feature_importances_"):
                importances.append(inner_clf.feature_importances_)

            print(
                f"  split {split_idx} seed {seed}: acc={m['accuracy']:.4f} "
                f"bacc={m['balanced_accuracy']:.4f} MAE={m['mae']:.4f} "
                f"tau={m['kendall_tau']:.4f} (inner {scoring}={search.best_score_:.4f})"
            )
        print(f"    chosen: {chosen[-1]}")

    metrics = summarize(
        f"{model_name} (tuned on {scoring}, {n_iter} samples x {INNER_FOLDS}-fold inner CV, "
        f"{len(effective_seeds)} seed(s))", fold_metrics
    )
    metrics["seeds"] = [int(s) for s in effective_seeds]
    metrics["chosen_params_per_fold"] = chosen
    metrics["predictions"] = predictions
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
            fold_metrics.append(_with_macro(y_test.to_numpy(), np.asarray(y_pred)))

        results[name] = summarize(name, fold_metrics)

    return results


# --------------------------------------------------------------------------------
# Untuned reference (reproduces guitar/baselines.py settings under this protocol)
# --------------------------------------------------------------------------------
def run_untuned_reference(features, splits, columns, seeds=(SEED,)):
    """The old hardcoded settings, evaluated identically, so the tuning delta is isolated."""
    results = {}
    definitions = {
        "random_forest_untuned": lambda s: RandomForestClassifier(random_state=s, n_estimators=200),
        "decision_tree_untuned": lambda s: DecisionTreeClassifier(random_state=s, max_depth=6),
    }
    for name, factory in definitions.items():
        fold_metrics = []
        for split_idx in range(N_SPLITS):
            X_pool, y_pool, X_test, y_test = fold_matrices(features, splits, split_idx)
            for seed in seeds:
                model = factory(seed).fit(X_pool, y_pool)
                y_pred = model.predict(X_test)
                fold_metrics.append(_with_macro(y_test.to_numpy(), y_pred))
        results[name] = summarize(name, fold_metrics)
        results[name]["seeds"] = [int(s) for s in seeds]
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
    parser.add_argument(
        "--seeds",
        default=str(SEED),
        help=(
            "comma-separated training seeds for the stochastic models, so their spread "
            "covers the same variation as RubricNet's 3 seeds x 5 folds (e.g. '0,1,2'). "
            "Deterministic models ignore this."
        ),
    )
    parser.add_argument("--skip-trivial", action="store_true")
    parser.add_argument(
        "--scoring",
        default=SELECTION_SCORING,
        help=(
            "inner-CV selection criterion. Comma-separate to run the search once per "
            "criterion (e.g. 'balanced_accuracy,accuracy'), which shows the baseline is "
            "not disadvantaged by the choice of tuning target. Besides sklearn's own "
            "names, 'macro_mae' and 'macro_mse' are accepted."
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

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    print(f"Seeds       : {list(seeds)} (stochastic models only)")

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
            "seeds": list(seeds),
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
    results.update(run_untuned_reference(features, splits, columns, seeds=seeds))

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
            results[key] = run_tuned(features, splits, columns, model_name, args.n_iter,
                                     scoring=scoring, seeds=seeds)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
