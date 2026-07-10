"""Fuzzy rule-based classifiers for guitar difficulty estimation, adapted from
Heerde, Vatolkin & Rudolph, "Comparing Fuzzy Rule Based Approaches for Music
Genre Classification" (EvoMUSART 2020) -- see fuzzy.txt at the repo root.

Two of the paper's three approaches are implemented (the evolutionary
approach is intentionally omitted):

- CompleteSearchClassifier: complete search of primitive rules
  "If <feature> is <term> then <class>", ranked by the relevance measure
  R(C, X, T) = P(X is T | C) * (1 - P(X is T))  (paper Eq. 1), following
  Vatolkin & Rudolph 2015.
- FuzzyPatternTreeClassifier: deterministic top-down fuzzy pattern tree
  induction (Senge & Huellermeier 2011/2015), one tree per class, greedily
  extending a leaf into an operator with a new statement child, minimizing
  RMSE against the one-vs-all 0/1 target.

Both operate one-vs-all over the classes and predict argmax of the mean (CS)
or direct (FPT) degree of truth -- faithful to the paper's nominal treatment,
even though the guitar task is ordinal. No seeds/repetitions are needed:
both methods are fully deterministic given the training data.
"""
import numpy as np

TERMS = ["very low", "low", "moderate", "high", "very high"]
CENTERS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
N_TERMS = len(TERMS)


def triangular_memberships(X01):
    """Fuzzify normalized features into 5 evenly spaced triangular terms.

    X01: (n, F) array with values in [0, 1] (already clipped by caller).
    Returns: (n, F, 5) array; memberships of adjacent terms sum to 1 at any x.
    """
    X01 = np.asarray(X01, dtype=np.float64)
    diff = np.abs(X01[:, :, None] - CENTERS[None, None, :])
    return np.clip(1.0 - diff / 0.25, 0.0, 1.0)


class Fuzzifier:
    """Normalizes raw feature values to [0, 1] before triangular fuzzification.

    method="cdf": per-feature empirical CDF of the training fold (robust to
    heavy-tailed descriptors like total_notes); values outside the observed
    train range clip to 0/1.
    method="minmax": plain (x - min) / (max - min), also clipped.
    """

    def __init__(self, method="cdf"):
        if method not in ("cdf", "minmax"):
            raise ValueError(f"unknown normalization method {method!r}")
        self.method = method
        self._sorted_train = None
        self._min = None
        self._max = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.method == "cdf":
            self._sorted_train = np.sort(X, axis=0)
        else:
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)
        return self

    def _normalize(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.method == "cdf":
            n = self._sorted_train.shape[0]
            ranks = np.empty_like(X)
            for j in range(X.shape[1]):
                col = self._sorted_train[:, j]
                if col[0] == col[-1]:
                    ranks[:, j] = 0.5
                else:
                    xp = col
                    fp = (np.arange(n) + 0.5) / n
                    ranks[:, j] = np.interp(X[:, j], xp, fp)
            return np.clip(ranks, 0.0, 1.0)
        span = self._max - self._min
        span = np.where(span == 0, 1.0, span)
        return np.clip((X - self._min) / span, 0.0, 1.0)

    def transform(self, X):
        return triangular_memberships(self._normalize(X))


class CompleteSearchClassifier:
    """Complete search of primitive fuzzy rules, one-vs-all per class."""

    def __init__(self, m, n_classes=8):
        self.m = m
        self.n_classes = n_classes
        self._order = None  # per class: list of (feat_idx, term_idx) sorted by relevance desc
        self._relevance = None

    def fit(self, M, y):
        """M: (n, F, 5) membership tensor. y: (n,) int class labels in [0, n_classes)."""
        n, n_feat, n_terms = M.shape
        y = np.asarray(y)
        p_t = M.mean(axis=0)  # (F, 5): P(X is T)
        self._order = []
        self._relevance = []
        for c in range(self.n_classes):
            mask = y == c
            if mask.sum() == 0:
                p_t_given_c = np.zeros((n_feat, n_terms))
            else:
                p_t_given_c = M[mask].mean(axis=0)
            relevance = p_t_given_c * (1.0 - p_t)  # (F, 5), Eq. 1
            flat = relevance.reshape(-1)
            # stable sort, descending relevance; ties broken by (feat_idx, term_idx) ascending
            idx_sorted = np.argsort(-flat, kind="stable")
            order = [(int(i // n_terms), int(i % n_terms)) for i in idx_sorted]
            self._order.append(order)
            self._relevance.append(flat[idx_sorted])
        return self

    def scores(self, M):
        """Mean truth of the top-m rules per class. Returns (n, n_classes)."""
        n = M.shape[0]
        out = np.zeros((n, self.n_classes))
        m = self.m
        for c in range(self.n_classes):
            order = self._order[c][:m]
            if not order:
                continue
            truths = np.stack([M[:, f, t] for f, t in order], axis=1)  # (n, m)
            out[:, c] = truths.mean(axis=1)
        return out

    def predict(self, M):
        return np.argmax(self.scores(M), axis=1)

    def rules_for_class(self, c, top_k, feature_names):
        order = self._order[c][:top_k]
        relevance = self._relevance[c][:top_k]
        rules = []
        for (f, t), r in zip(order, relevance):
            rules.append({
                "feature": feature_names[f],
                "term": TERMS[t],
                "relevance": float(r),
                "text": f"IF {feature_names[f]} IS {TERMS[t]} THEN class {c}",
            })
        return rules

    @staticmethod
    def sweep_m(M_train, y_train, M_val, y_val, n_classes=8, max_m=None):
        """Fit once on train, evaluate val balanced accuracy for every m in
        1..max_m (default: all F*5 rules) via cumulative means. Returns dict
        {m: balanced_accuracy}."""
        from sklearn.metrics import balanced_accuracy_score

        n_feat = M_train.shape[1]
        n_terms = M_train.shape[2]
        pool_size = n_feat * n_terms
        if max_m is None:
            max_m = pool_size

        clf = CompleteSearchClassifier(m=pool_size, n_classes=n_classes).fit(M_train, y_train)

        n_val = M_val.shape[0]
        # cumulative_truth[c] has shape (n_val, pool_size): running mean of top-k rule truths
        results = {}
        cum_scores = np.zeros((n_val, n_classes, max_m))
        for c in range(n_classes):
            order = clf._order[c][:max_m]
            if not order:
                continue
            truths = np.stack([M_val[:, f, t] for f, t in order], axis=1)  # (n_val, max_m)
            running_sum = np.cumsum(truths, axis=1)
            counts = np.arange(1, truths.shape[1] + 1)
            cum_scores[:, c, :truths.shape[1]] = running_sum / counts[None, :]
            if truths.shape[1] < max_m:
                cum_scores[:, c, truths.shape[1]:] = cum_scores[:, c, truths.shape[1] - 1:truths.shape[1]]

        for m in range(1, max_m + 1):
            preds = np.argmax(cum_scores[:, :, m - 1], axis=1)
            results[m] = balanced_accuracy_score(y_val, preds)
        return results, clf


# --- Fuzzy Pattern Trees -----------------------------------------------

class FPTNode:
    """Leaf: op is None, (feat_idx, term_idx, negated) set.
    Inner: op in {"and", "or", "avg"}, children is a list of 2 FPTNode."""

    __slots__ = ("op", "feat_idx", "term_idx", "negated", "children")

    def __init__(self, op=None, feat_idx=None, term_idx=None, negated=False, children=None):
        self.op = op
        self.feat_idx = feat_idx
        self.term_idx = term_idx
        self.negated = negated
        self.children = children

    @property
    def is_leaf(self):
        return self.op is None

    def eval(self, M):
        """M: (n, F, 5) -> (n,) degree of truth."""
        if self.is_leaf:
            mu = M[:, self.feat_idx, self.term_idx]
            return 1.0 - mu if self.negated else mu
        a = self.children[0].eval(M)
        b = self.children[1].eval(M)
        if self.op == "and":
            return np.minimum(a, b)
        if self.op == "or":
            return np.maximum(a, b)
        return 0.5 * (a + b)  # avg

    def expression(self, feature_names):
        if self.is_leaf:
            stmt = f"{feature_names[self.feat_idx]} IS {TERMS[self.term_idx]}"
            return f"NOT ({stmt})" if self.negated else stmt
        op_name = {"and": "AND", "or": "OR", "avg": "AVG"}[self.op]
        left = self.children[0].expression(feature_names)
        right = self.children[1].expression(feature_names)
        return f"{op_name}({left}, {right})"

    def to_dict(self, feature_names):
        if self.is_leaf:
            return {
                "feature": feature_names[self.feat_idx],
                "term": TERMS[self.term_idx],
                "negated": self.negated,
            }
        return {
            "op": self.op,
            "children": [c.to_dict(feature_names) for c in self.children],
        }

    def leaves(self):
        if self.is_leaf:
            return [self]
        out = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def depth(self):
        if self.is_leaf:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def copy_with_leaf_replaced(self, target, replacement):
        """Return a new tree with `target` (identity match) swapped for `replacement`."""
        if self is target:
            return replacement
        if self.is_leaf:
            return FPTNode(op=None, feat_idx=self.feat_idx, term_idx=self.term_idx, negated=self.negated)
        new_children = [c.copy_with_leaf_replaced(target, replacement) for c in self.children]
        return FPTNode(op=self.op, children=new_children)


def _all_statements(n_feat, n_terms, use_negation):
    """List of (feat_idx, term_idx, negated) candidate leaf statements."""
    stmts = [(f, t, False) for f in range(n_feat) for t in range(n_terms)]
    if use_negation:
        stmts += [(f, t, True) for f in range(n_feat) for t in range(n_terms)]
    return stmts


def _weighted_rmse(y_true, y_pred, weights):
    err = (y_true - y_pred) ** 2
    return float(np.sqrt(np.sum(weights * err)))


class FuzzyPatternTreeClassifier:
    """One deterministic fuzzy pattern tree per class, top-down induction."""

    def __init__(self, d_max, gamma=0.05, use_negation=True, balanced_rmse=True, n_classes=8):
        self.d_max = d_max
        self.gamma = gamma
        self.use_negation = use_negation
        self.balanced_rmse = balanced_rmse
        self.n_classes = n_classes
        self._trees = None
        self.negation_used_ = False

    def _sample_weights(self, y_bin):
        if not self.balanced_rmse:
            n = len(y_bin)
            return np.full(n, 1.0 / n)
        n_pos = int(y_bin.sum())
        n_neg = len(y_bin) - n_pos
        w = np.empty(len(y_bin))
        w[y_bin == 1] = 1.0 / (2 * n_pos) if n_pos > 0 else 0.0
        w[y_bin == 0] = 1.0 / (2 * n_neg) if n_neg > 0 else 0.0
        return w

    def _induce_one(self, M, y_bin):
        n, n_feat, n_terms = M.shape
        weights = self._sample_weights(y_bin)
        stmts = _all_statements(n_feat, n_terms, self.use_negation)

        stmt_outputs = np.stack(
            [(1.0 - M[:, f, t]) if neg else M[:, f, t] for f, t, neg in stmts], axis=1
        )  # (n, n_stmts)

        # Initialize: best single statement.
        errs = np.array([
            _weighted_rmse(y_bin, stmt_outputs[:, i], weights) for i in range(len(stmts))
        ])
        best_i = int(np.argmin(errs))
        f0, t0, neg0 = stmts[best_i]
        tree = FPTNode(op=None, feat_idx=f0, term_idx=t0, negated=neg0)
        current_err = errs[best_i]

        if self.d_max <= 1:
            return tree

        while True:
            if tree.depth() >= self.d_max:
                break
            leaves = tree.leaves()
            # only leaves whose depth-from-root allows one more level are extendable
            extendable = [lf for lf in leaves if self._leaf_depth(tree, lf) < self.d_max]
            if not extendable:
                break

            best_candidate = None
            best_candidate_err = current_err

            for leaf in extendable:
                leaf_out = leaf.eval(M)
                for op in ("and", "or", "avg"):
                    # vectorized over all candidate new statements
                    if op == "and":
                        combined = np.minimum(leaf_out[:, None], stmt_outputs)
                    elif op == "or":
                        combined = np.maximum(leaf_out[:, None], stmt_outputs)
                    else:
                        combined = 0.5 * (leaf_out[:, None] + stmt_outputs)

                    # Replacing `leaf` only changes the tree's output along the
                    # leaf-to-root path; for non-root leaves the rest of the
                    # tree must be folded back in, done by _score_candidates.
                    cand_full_outputs = self._score_candidates(tree, leaf, combined, M)
                    cand_errs = np.array([
                        _weighted_rmse(y_bin, cand_full_outputs[:, i], weights)
                        for i in range(cand_full_outputs.shape[1])
                    ])
                    local_best = int(np.argmin(cand_errs))
                    if cand_errs[local_best] < best_candidate_err:
                        best_candidate_err = cand_errs[local_best]
                        f, t, neg = stmts[local_best]
                        new_leaf = FPTNode(
                            op=op,
                            children=[
                                FPTNode(op=None, feat_idx=leaf.feat_idx, term_idx=leaf.term_idx, negated=leaf.negated),
                                FPTNode(op=None, feat_idx=f, term_idx=t, negated=neg),
                            ],
                        )
                        best_candidate = (leaf, new_leaf)

            if best_candidate is None:
                break
            if best_candidate_err > (1.0 + self.gamma) * current_err:
                break

            leaf, new_leaf = best_candidate
            tree = tree.copy_with_leaf_replaced(leaf, new_leaf)
            current_err = best_candidate_err

        return tree

    @staticmethod
    def _leaf_depth(root, target_leaf):
        """Depth (1-indexed, root=1) of target_leaf within root."""
        def _walk(node, d):
            if node is target_leaf:
                return d
            if node.is_leaf:
                return None
            for c in node.children:
                r = _walk(c, d + 1)
                if r is not None:
                    return r
            return None
        return _walk(root, 1)

    @staticmethod
    def _score_candidates(tree, target_leaf, combined, M):
        """Evaluate the whole tree's output for each candidate replacement of
        `target_leaf` with a subtree whose output is `combined[:, i]`.

        Implemented by re-evaluating the tree once per candidate is wasteful
        for deep trees; instead we exploit that eval() is a pure function of
        leaf outputs, so we evaluate the tree symbolically: replace the
        target leaf's contribution with each candidate column, propagating
        up through the (small, <=5-deep) path from leaf to root.
        """
        n = M.shape[0]
        n_cand = combined.shape[1]

        def _eval_with_override(node):
            """Returns (n,) or (n, n_cand) depending on whether target_leaf is
            in this subtree."""
            if node is target_leaf:
                return combined  # (n, n_cand)
            if node.is_leaf:
                mu = M[:, node.feat_idx, node.term_idx]
                return 1.0 - mu if node.negated else mu  # (n,)
            a = _eval_with_override(node.children[0])
            b = _eval_with_override(node.children[1])
            a_is_cand = a.ndim == 2
            b_is_cand = b.ndim == 2
            if not a_is_cand and not b_is_cand:
                if node.op == "and":
                    return np.minimum(a, b)
                if node.op == "or":
                    return np.maximum(a, b)
                return 0.5 * (a + b)
            a2 = a if a_is_cand else np.broadcast_to(a[:, None], (n, n_cand))
            b2 = b if b_is_cand else np.broadcast_to(b[:, None], (n, n_cand))
            if node.op == "and":
                return np.minimum(a2, b2)
            if node.op == "or":
                return np.maximum(a2, b2)
            return 0.5 * (a2 + b2)

        result = _eval_with_override(tree)
        if result.ndim == 1:
            result = np.broadcast_to(result[:, None], (n, n_cand))
        return result

    def fit(self, M, y):
        y = np.asarray(y)
        self._trees = []
        self.negation_used_ = False
        for c in range(self.n_classes):
            y_bin = (y == c).astype(np.float64)
            tree = self._induce_one(M, y_bin)
            for leaf in tree.leaves():
                if leaf.negated:
                    self.negation_used_ = True
            self._trees.append(tree)
        return self

    def scores(self, M):
        n = M.shape[0]
        out = np.zeros((n, self.n_classes))
        for c, tree in enumerate(self._trees):
            out[:, c] = tree.eval(M)
        return out

    def predict(self, M):
        return np.argmax(self.scores(M), axis=1)

    def is_constant(self, M, tol=1e-9):
        """True if any class tree is (near-)constant on M."""
        s = self.scores(M)
        return bool(np.any(s.max(axis=0) - s.min(axis=0) < tol))

    def tree_expression(self, c, feature_names):
        return self._trees[c].expression(feature_names)

    def tree_dict(self, c, feature_names):
        return self._trees[c].to_dict(feature_names)
