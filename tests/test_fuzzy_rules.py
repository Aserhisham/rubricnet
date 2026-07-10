import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from guitar.fuzzy_rules import (
    CENTERS, CompleteSearchClassifier, Fuzzifier, FuzzyPatternTreeClassifier,
    triangular_memberships,
)


def test_partition_of_unity():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 3))
    mu = triangular_memberships(x)
    assert mu.shape == (200, 3, 5)
    assert np.all(mu >= 0) and np.all(mu <= 1)
    # every point lies in at most 2 adjacent triangles, which always sum to 1
    assert np.allclose(mu.sum(axis=2), 1.0, atol=1e-9)


def test_one_hot_at_centers():
    # one row, 5 features, feature j's value is exactly the center of term j
    x = CENTERS[None, :]
    mu = triangular_memberships(x)
    # mu shape (1, 5, 5): feature j evaluated at CENTERS[j] should be one-hot at term j
    for j in range(5):
        row = mu[0, j]
        assert np.argmax(row) == j
        assert row[j] == pytest.approx(1.0)


def test_fuzzifier_cdf_monotone_and_clips():
    rng = np.random.default_rng(1)
    x_train = rng.exponential(scale=2.0, size=(300, 1))
    fz = Fuzzifier(method="cdf").fit(x_train)
    x_test = np.array([[-5.0], [0.0], [x_train.mean()], [1e6]])
    mu = fz.transform(x_test)
    normalized = fz._normalize(x_test)
    # values outside the observed train range clamp near the 0/1 boundary
    # (continuity-corrected ECDF: (rank + 0.5) / n, never exactly 0 or 1)
    assert normalized[0, 0] < 0.01  # far below range
    assert normalized[-1, 0] > 0.99  # far above range
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))
    # monotonicity property: larger raw x -> larger or equal normalized value
    xs = np.sort(rng.uniform(x_train.min(), x_train.max(), size=50))[:, None]
    normed = fz._normalize(xs)[:, 0]
    assert np.all(np.diff(normed) >= -1e-9)


def test_fuzzifier_minmax():
    x_train = np.array([[0.0], [10.0]])
    fz = Fuzzifier(method="minmax").fit(x_train)
    out = fz._normalize(np.array([[-5.0], [0.0], [5.0], [10.0], [15.0]]))
    assert np.allclose(out[:, 0], [0.0, 0.0, 0.5, 1.0, 1.0])


def test_complete_search_separable():
    rng = np.random.default_rng(2)
    n_classes = 4
    n = 400
    y = rng.integers(0, n_classes, size=n)
    # feature 0 encodes the class almost perfectly; feature 1 is noise
    x0 = y / (n_classes - 1) + rng.normal(0, 0.02, size=n)
    x1 = rng.uniform(0, 1, size=n)
    X = np.clip(np.stack([x0, x1], axis=1), 0, 1)
    M = triangular_memberships(X)

    clf = CompleteSearchClassifier(m=1, n_classes=n_classes).fit(M, y)
    preds = clf.predict(M)
    acc = (preds == y).mean()
    assert acc > 0.85

    # the most relevant rule for every class should reference feature 0
    for c in range(n_classes):
        assert clf._order[c][0][0] == 0


def test_sweep_m_matches_direct_fit():
    rng = np.random.default_rng(3)
    n_classes = 3
    X_train = rng.uniform(0, 1, size=(150, 4))
    y_train = rng.integers(0, n_classes, size=150)
    X_val = rng.uniform(0, 1, size=(60, 4))
    y_val = rng.integers(0, n_classes, size=60)
    M_train = triangular_memberships(X_train)
    M_val = triangular_memberships(X_val)

    results, clf = CompleteSearchClassifier.sweep_m(M_train, y_train, M_val, y_val, n_classes=n_classes)
    assert set(results.keys()) == set(range(1, 4 * 5 + 1))

    for m in (1, 5, 12, 20):
        direct = CompleteSearchClassifier(m=m, n_classes=n_classes).fit(M_train, y_train)
        from sklearn.metrics import balanced_accuracy_score
        expected = balanced_accuracy_score(y_val, direct.predict(M_val))
        assert results[m] == pytest.approx(expected)


def test_fpt_dmax1_is_best_single_statement():
    rng = np.random.default_rng(4)
    n = 200
    X = rng.uniform(0, 1, size=(n, 3))
    y = (X[:, 0] > 0.5).astype(int)  # binary "class"
    M = triangular_memberships(X)

    fpt = FuzzyPatternTreeClassifier(d_max=1, n_classes=2, balanced_rmse=False).fit(M, y)
    tree0 = fpt._trees[1]
    assert tree0.is_leaf

    # brute-force best single statement for class 1
    from guitar.fuzzy_rules import _all_statements, _weighted_rmse
    stmts = _all_statements(3, 5, True)
    y_bin = y.astype(np.float64)
    w = np.full(n, 1.0 / n)
    best_err = min(
        _weighted_rmse(y_bin, (1.0 - M[:, f, t]) if neg else M[:, f, t], w)
        for f, t, neg in stmts
    )
    achieved_err = _weighted_rmse(y_bin, tree0.eval(M), w)
    assert achieved_err == pytest.approx(best_err)


def test_fpt_error_never_worsens_beyond_gamma():
    rng = np.random.default_rng(5)
    n = 150
    X = rng.uniform(0, 1, size=(n, 5))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    M = triangular_memberships(X)
    gamma = 0.05

    for d_max in (1, 2, 3, 4, 5):
        fpt = FuzzyPatternTreeClassifier(d_max=d_max, gamma=gamma, n_classes=2).fit(M, y)
        # depth constraint respected
        assert fpt._trees[1].depth() <= d_max


def test_balanced_rmse_constant_tree_is_half():
    from guitar.fuzzy_rules import _weighted_rmse
    n_pos, n_neg = 30, 170
    y = np.array([1.0] * n_pos + [0.0] * n_neg)
    w = np.empty(len(y))
    w[:n_pos] = 1.0 / (2 * n_pos)
    w[n_pos:] = 1.0 / (2 * n_neg)
    pred = np.full(len(y), 0.5)
    assert _weighted_rmse(y, pred, w) == pytest.approx(0.5)


def test_determinism():
    rng = np.random.default_rng(6)
    n_classes = 5
    X = rng.uniform(0, 1, size=(300, 6))
    y = rng.integers(0, n_classes, size=300)
    M = triangular_memberships(X)

    cs1 = CompleteSearchClassifier(m=10, n_classes=n_classes).fit(M, y)
    cs2 = CompleteSearchClassifier(m=10, n_classes=n_classes).fit(M, y)
    assert cs1._order == cs2._order
    assert np.array_equal(cs1.predict(M), cs2.predict(M))

    fpt1 = FuzzyPatternTreeClassifier(d_max=3, n_classes=n_classes).fit(M, y)
    fpt2 = FuzzyPatternTreeClassifier(d_max=3, n_classes=n_classes).fit(M, y)
    feature_names = [f"f{i}" for i in range(6)]
    for c in range(n_classes):
        assert fpt1.tree_expression(c, feature_names) == fpt2.tree_expression(c, feature_names)
    assert np.array_equal(fpt1.predict(M), fpt2.predict(M))
