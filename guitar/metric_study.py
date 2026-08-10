"""Which evaluation metrics are appropriate for this imbalanced ordinal task?

Motivation
----------
The thesis reports accuracy, balanced accuracy, acc+/-1, MAE, MSE and Kendall's tau.
Two properties of this dataset make that suite incomplete:

  * **Imbalance.** The 640-piece V5 set has class sizes 105/96/124/80/79/70/55/31
    (4.0x between the largest and smallest). The three easiest classes hold 325 of
    640 pieces (50.8%), so plain accuracy and plain MAE are majority-class metrics:
    a model can improve them while getting *worse* on the hard end of the scale.
  * **Ordinality.** Accuracy and balanced accuracy discard the class order entirely;
    a 1-bin miss and a 7-bin miss are equally wrong to them.

Only *one* metric currently reported (balanced accuracy) corrects for imbalance, and
it is precisely the one that ignores order. No reported metric does both. This script
adds the two standard measures that do, evaluates every metric's behaviour on this
specific label distribution, and produces the evidence for a thesis subsection
recommending a primary metric.

Metrics added
-------------
  * **Macro-averaged MAE (MAE^M)** -- Baccianella, Esuli & Sebastiani (2009), the
    reference measure for ordinal regression under imbalance: per-class MAE, then
    averaged unweighted over classes. Order-aware *and* imbalance-corrected. This is
    the single most defensible primary error metric for this task.
  * **Quadratic / linear weighted kappa** -- Cohen's kappa with ordinal penalty
    weights, chance-corrected. Answers "how much better than guessing the marginal
    distribution", which raw accuracy cannot: accuracy of 0.334 sounds weak until
    placed against the 0.125 uniform / ~0.152 marginal chance level.
  * **Kendall tau-c** alongside the currently reported tau-b. With 8 ordinal classes
    and 128 test pieces there are massive ties; tau-b's tie correction assumes both
    margins are comparable, while tau-c is the variant intended for non-square tables.
  * **Spearman rho** as a second, tie-tolerant rank check.

Diagnostics produced
--------------------
  1. Chance-level table: what each metric scores under majority / marginal-stratified /
     uniform-random prediction, so every headline number can be read against its floor.
  2. Class-size sensitivity: how much each metric's value is determined by the three
     largest classes, computed by recomputing each metric on a class-balanced resample.
  3. Bootstrap 95% CIs over pieces, showing which reported differences are resolvable
     at this sample size at all.
  4. Model ranking under each metric, to show whether the metric choice changes the
     conclusion (the practically important question).

Usage
-----
    python -m guitar.metric_study                 # recover predictions + full report
    python -m guitar.metric_study --report-only   # reuse cached predictions
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

os.environ.setdefault("WANDB_MODE", "disabled")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR2, NUM_CLASSES
from guitar.train_guitar_rubricnet import DEFAULT_HYPERPARAMS
from rubricnet.rubricnet import RubricnetSklearn

CSV_PATH = "features/guitar_descriptors_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
COLUMNS = ALL_FEATURES_V5_PRUNED_COLLINEAR2
RUBRICNET_ALIAS = "guitar_rubricnet_final_v5_pruned_collinear2"
BEST_HYPERPARAMS_PATH = "guitar/best_hyperparams_guitar_all_v5.json"
CKPT_DIR = "checkpoints"
PRED_CACHE = "guitar/metric_study_predictions.json"
REPORT_PATH = "guitar/METRIC_STUDY.md"
SEEDS = [0, 1, 2]
LEVEL_RANGES = ["1-3", "4-5", "6-7", "8", "9-10", "11-12", "13-15", "16-20"]
RNG = np.random.default_rng(0)


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# --------------------------------------------------------------------------------
# Metric definitions
# --------------------------------------------------------------------------------
def macro_mae(y_true, y_pred, num_classes=NUM_CLASSES):
    """Macro-averaged MAE (MAE^M), Baccianella et al. (2009).

    Per-class MAE averaged unweighted across the classes actually present. Unlike
    plain MAE, every difficulty level contributes equally regardless of how many
    pieces it holds, so a model cannot buy a good score by being accurate only on
    the well-populated easy end of the scale.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_class = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        per_class[c] = float(np.abs(y_pred[mask] - y_true[mask]).mean())
    return float(np.mean(list(per_class.values()))), per_class


def macro_mse(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    vals = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum():
            vals.append(float(((y_pred[mask] - y_true[mask]) ** 2).mean()))
    return float(np.mean(vals))


def kendall_tau_c(y_true, y_pred):
    """Stuart's tau-c. REPORTED FOR DIAGNOSIS ONLY -- not safe for model comparison here.

    tau-c is proportional to m/(m-1), where m is the smaller of the number of distinct
    true and *predicted* classes. On this task the models differ in how many distinct
    classes they emit: RubricNet never predicts class 7 (m=7) while the Random Forest
    does (m=8). That difference alone multiplies RubricNet's tau-c by (8/7)/(7/6) ~ 1.021
    relative to the forest -- inflating the score of the model that is *worse* on the
    rarest class, purely because it declines to predict it.

    Measured on this data: RubricNet scores tau-c 0.6309 against the forest's 0.6039, but
    rescaled to a common m=8 footing it is 0.6180 -- roughly half the apparent gap is the
    normalisation artefact rather than model quality. Use tau-b (which does not depend on
    m) or macro-MAE for comparisons; keep tau-c only as a descriptive statistic.
    """
    res = kendalltau(y_true, y_pred, variant="c")
    tau = res.correlation if hasattr(res, "correlation") else res[0]
    return 0.0 if np.isnan(tau) else float(tau)


def n_distinct_predicted(y_true, y_pred):
    """m as used by tau-c: min(#distinct true, #distinct predicted)."""
    return min(len(set(np.asarray(y_true).tolist())), len(set(np.asarray(y_pred).tolist())))


def kendall_tau_b(y_true, y_pred):
    res = kendalltau(y_true, y_pred)
    tau = res.correlation if hasattr(res, "correlation") else res[0]
    return 0.0 if np.isnan(tau) else float(tau)


def spearman_rho(y_true, y_pred):
    rho = spearmanr(y_true, y_pred).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def all_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        mean_absolute_error,
        mean_squared_error,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    mmae, _ = macro_mae(y_true, y_pred, num_classes)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "acc_plus_minus_1": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "macro_mae": mmae,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "macro_mse": macro_mse(y_true, y_pred, num_classes),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        "lwk": float(cohen_kappa_score(y_true, y_pred, weights="linear", labels=labels)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "kendall_tau_b": kendall_tau_b(y_true, y_pred),
        "kendall_tau_c": kendall_tau_c(y_true, y_pred),
        "spearman_rho": spearman_rho(y_true, y_pred),
    }


# Direction each metric should move for a better model.
HIGHER_IS_BETTER = {
    "accuracy": True, "balanced_accuracy": True, "acc_plus_minus_1": True,
    "mae": False, "macro_mae": False, "mse": False, "macro_mse": False,
    "qwk": True, "lwk": True, "kappa": True,
    "kendall_tau_b": True, "kendall_tau_c": True, "spearman_rho": True,
}


# --------------------------------------------------------------------------------
# Prediction recovery
# --------------------------------------------------------------------------------
def load_hyperparams(path):
    with open(path) as f:
        tuned = json.load(f)["params"]
    hp = dict(DEFAULT_HYPERPARAMS)
    hp.update(tuned)
    return hp


def recover_predictions():
    """Re-derive test predictions per fold for RubricNet (3 seeds) and the RF baseline."""
    features, splits = load_data(csv_path=CSV_PATH, splits_path=SPLITS_PATH, columns=COLUMNS)
    hyperparams = load_hyperparams(BEST_HYPERPARAMS_PATH)
    records = []

    for seed in SEEDS:
        alias = f"{RUBRICNET_ALIAS}_seed_{seed}"
        for split_idx in range(5):
            X_train, _ = get_fold_xy(features, splits, split_idx, "train")
            X_test, y_test = get_fold_xy(features, splits, split_idx, "test")
            medians = X_train.median().fillna(0.0)
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)
            scaler = StandardScaler().fit(X_train)

            clf = RubricnetSklearn(
                input_dim=len(COLUMNS), num_classes=NUM_CLASSES, split=split_idx,
                args=Args(alias_experiment=alias, **hyperparams), logging=False,
            )
            clf.load_model(f"{CKPT_DIR}/{alias}/split_{split_idx}.ckpt")
            y_pred = np.clip(clf.predict(scaler.transform(X_test)).cpu().numpy(), 0, NUM_CLASSES - 1)

            records.append({
                "model": "RubricNet", "seed": seed, "fold": split_idx,
                "y_true": y_test.to_numpy().tolist(), "y_pred": [int(v) for v in y_pred],
            })
            print(f"  RubricNet seed {seed} fold {split_idx}: {len(y_pred)} predictions")

    # Random Forest under the thesis's own protocol (val folded back into train).
    for split_idx in range(5):
        X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
        X_val, y_val = get_fold_xy(features, splits, split_idx, "val")
        X_test, y_test = get_fold_xy(features, splits, split_idx, "test")
        medians = X_train.median().fillna(0.0)
        import pandas as pd
        X_pool = pd.concat([X_train.fillna(medians), X_val.fillna(medians)])
        y_pool = pd.concat([y_train, y_val])
        rf = RandomForestClassifier(random_state=42, n_estimators=200).fit(X_pool, y_pool)
        y_pred = rf.predict(X_test.fillna(medians))
        records.append({
            "model": "RandomForest", "seed": 0, "fold": split_idx,
            "y_true": y_test.to_numpy().tolist(), "y_pred": [int(v) for v in y_pred],
        })
        print(f"  RandomForest fold {split_idx}: {len(y_pred)} predictions")

    with open(PRED_CACHE, "w") as f:
        json.dump(records, f)
    print(f"\nCached predictions -> {PRED_CACHE}")
    return records


# --------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------
def pooled(records, model):
    yt, yp = [], []
    for r in records:
        if r["model"] == model:
            yt.extend(r["y_true"])
            yp.extend(r["y_pred"])
    return np.array(yt), np.array(yp)


def per_fold_metrics(records, model):
    out = []
    for r in records:
        if r["model"] == model:
            out.append(all_metrics(np.array(r["y_true"]), np.array(r["y_pred"])))
    return out


def chance_levels(y_true, n_draws=200):
    """What each metric scores for three reference predictors that have learned nothing."""
    y_true = np.asarray(y_true)
    counts = Counter(y_true.tolist())
    majority = counts.most_common(1)[0][0]
    classes = np.array(sorted(counts))
    probs = np.array([counts[c] for c in classes], dtype=float)
    probs /= probs.sum()

    rows = {"majority_class": all_metrics(y_true, np.full_like(y_true, majority))}

    for name, p in (("marginal_random", probs), ("uniform_random", np.full(len(classes), 1 / len(classes)))):
        acc = []
        for _ in range(n_draws):
            yp = RNG.choice(classes, size=len(y_true), p=p)
            acc.append(all_metrics(y_true, yp))
        rows[name] = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}

    # The best constant predictor under MAE is the median class, which differs from
    # the majority class and is the correct floor for the error metrics.
    median_cls = int(np.median(y_true))
    rows["median_class"] = all_metrics(y_true, np.full_like(y_true, median_cls))
    return rows


def balanced_resample_metrics(y_true, y_pred, n_draws=400):
    """Recompute every metric on a class-balanced resample of the test pieces.

    The gap between a metric's value on the natural distribution and on the balanced
    resample is exactly that metric's exposure to the class imbalance: a metric that
    moves a lot is being driven by the over-represented easy classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true.tolist()))
    idx_by_class = {c: np.flatnonzero(y_true == c) for c in classes}
    n_per_class = min(len(v) for v in idx_by_class.values())

    draws = []
    for _ in range(n_draws):
        idx = np.concatenate([RNG.choice(idx_by_class[c], n_per_class, replace=False) for c in classes])
        draws.append(all_metrics(y_true[idx], y_pred[idx]))
    return {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}


def bootstrap_ci(y_true, y_pred, n_boot=2000, alpha=0.05):
    """Percentile bootstrap CI over pieces, for every metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    draws = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        draws.append(all_metrics(y_true[idx], y_pred[idx]))
    out = {}
    for k in draws[0]:
        vals = np.array([d[k] for d in draws])
        out[k] = (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))
    return out


def paired_bootstrap_difference(yt_a, yp_a, yp_b, n_boot=2000):
    """Paired bootstrap on the same pieces: P(model A better than model B) per metric.

    Requires both models evaluated on identical y_true ordering.
    """
    yt_a = np.asarray(yt_a)
    n = len(yt_a)
    wins = Counter()
    total = 0
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(set(yt_a[idx].tolist())) < 2:
            continue
        ma = all_metrics(yt_a[idx], np.asarray(yp_a)[idx])
        mb = all_metrics(yt_a[idx], np.asarray(yp_b)[idx])
        total += 1
        for k in ma:
            better = ma[k] > mb[k] if HIGHER_IS_BETTER[k] else ma[k] < mb[k]
            if better:
                wins[k] += 1
    return {k: wins[k] / total for k in HIGHER_IS_BETTER}, total


# --------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------
METRIC_ORDER = [
    "accuracy", "balanced_accuracy", "acc_plus_minus_1",
    "mae", "macro_mae", "mse", "macro_mse",
    "kappa", "lwk", "qwk",
    "kendall_tau_b", "kendall_tau_c", "spearman_rho",
]

METRIC_LABEL = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "acc_plus_minus_1": "Accuracy +/-1", "mae": "MAE", "macro_mae": "Macro-MAE (MAE^M)",
    "mse": "MSE", "macro_mse": "Macro-MSE", "kappa": "Cohen kappa",
    "lwk": "Linear weighted kappa", "qwk": "Quadratic weighted kappa",
    "kendall_tau_b": "Kendall tau-b", "kendall_tau_c": "Kendall tau-c",
    "spearman_rho": "Spearman rho",
}


def fmt(v):
    return f"{v:.4f}"


def build_report(records):
    lines = []
    w = lines.append

    w("# Evaluation Metrics for an Imbalanced Ordinal Task\n")
    w("Generated by `guitar/metric_study.py`. All numbers on the 640-piece V5 dataset, ")
    w("25-descriptor V5-pruned-collinear2 set, pooled over the 5 test folds ")
    w("(RubricNet additionally over 3 seeds).\n")

    yt_r, yp_r = pooled(records, "RubricNet")
    yt_f, yp_f = pooled(records, "RandomForest")

    # ---- label distribution
    counts = Counter(yt_f.tolist())
    total = sum(counts.values())
    w("\n## 1. The label distribution that motivates the question\n")
    w("| Class | Grades | N | Share |")
    w("|---|---|---:|---:|")
    for c in sorted(counts):
        w(f"| {c} | {LEVEL_RANGES[c]} | {counts[c]} | {counts[c]/total:.1%} |")
    w(f"\nImbalance ratio (largest/smallest): **{max(counts.values())/min(counts.values()):.2f}x**. ")
    easy = sum(counts[c] for c in (0, 1, 2))
    w(f"Classes 0-2 hold **{easy}/{total} = {easy/total:.1%}** of all pieces, so any ")
    w("unweighted metric is predominantly a measurement of performance on easy repertoire.\n")

    # ---- chance levels
    w("\n## 2. Chance levels\n")
    w("What a model that has learned nothing scores. Reported figures must be read against these floors.\n")
    ch = chance_levels(yt_f)
    w("| Metric | Majority class | Median class | Marginal random | Uniform random |")
    w("|---|---:|---:|---:|---:|")
    for m in METRIC_ORDER:
        w(f"| {METRIC_LABEL[m]} | {fmt(ch['majority_class'][m])} | {fmt(ch['median_class'][m])} | "
          f"{fmt(ch['marginal_random'][m])} | {fmt(ch['uniform_random'][m])} |")
    w("\nNote the majority-class predictor scores **MAE " + fmt(ch["majority_class"]["mae"]) +
      "** but **macro-MAE " + fmt(ch["majority_class"]["macro_mae"]) + "** -- the ")
    w("degenerate model looks far better under the plain metric than the macro one, which ")
    w("is precisely the failure mode macro-averaging exists to expose.\n")

    # ---- model comparison
    w("\n## 3. Model comparison under every metric\n")
    mr = all_metrics(yt_r, yp_r)
    mf = all_metrics(yt_f, yp_f)
    w("| Metric | RubricNet | Random Forest | Better |")
    w("|---|---:|---:|:---:|")
    for m in METRIC_ORDER:
        better = "RubricNet" if (mr[m] > mf[m]) == HIGHER_IS_BETTER[m] else "RF"
        if abs(mr[m] - mf[m]) < 1e-9:
            better = "tie"
        w(f"| {METRIC_LABEL[m]} | {fmt(mr[m])} | {fmt(mf[m])} | {better} |")

    # ---- imbalance exposure
    w("\n## 4. How much each metric is driven by the imbalance\n")
    w("Each metric recomputed on class-balanced resamples of the same predictions ")
    w("(equal pieces per class). A large shift means the metric's headline value is ")
    w("substantially an artefact of the class distribution rather than of model quality.\n")
    bal_r = balanced_resample_metrics(yt_r, yp_r)
    w("| Metric | Natural distribution | Class-balanced | Shift |")
    w("|---|---:|---:|---:|")
    for m in METRIC_ORDER:
        w(f"| {METRIC_LABEL[m]} | {fmt(mr[m])} | {fmt(bal_r[m])} | {bal_r[m]-mr[m]:+.4f} |")

    # ---- resolution
    w("\n## 5. What this sample size can actually resolve\n")
    w("Percentile bootstrap 95% CIs over pieces for RubricNet, and a paired bootstrap ")
    w("giving the fraction of resamples in which RubricNet beats the Random Forest.\n")
    ci = bootstrap_ci(yt_r, yp_r)
    # Pair on the seed-0 RubricNet predictions so both models see identical pieces.
    r0 = [r for r in records if r["model"] == "RubricNet" and r["seed"] == 0]
    f0 = [r for r in records if r["model"] == "RandomForest"]
    yt_p = np.concatenate([np.array(r["y_true"]) for r in r0])
    yp_p_r = np.concatenate([np.array(r["y_pred"]) for r in r0])
    yp_p_f = np.concatenate([np.array(r["y_pred"]) for r in f0])
    winrate, n_used = paired_bootstrap_difference(yt_p, yp_p_r, yp_p_f)

    w("| Metric | RubricNet (95% CI) | CI width | P(RubricNet > RF) |")
    w("|---|---:|---:|---:|")
    for m in METRIC_ORDER:
        lo, hi = ci[m]
        w(f"| {METRIC_LABEL[m]} | {fmt(mr[m])} [{fmt(lo)}, {fmt(hi)}] | {hi-lo:.4f} | {winrate[m]:.3f} |")
    w(f"\nPaired bootstrap over {n_used} resamples of the {len(yt_p)} seed-0 test predictions.\n")

    # ---- tau-c caveat
    m_r = n_distinct_predicted(yt_p, yp_p_r)
    m_f = n_distinct_predicted(yt_p, yp_p_f)
    w("\n### Caveat: Kendall tau-c must not be used for model comparison here\n")
    w(f"RubricNet emits **{m_r}** distinct classes on this fold set; the Random Forest emits ")
    w(f"**{m_f}**. RubricNet never predicts class 7 at all. Since tau-c is proportional to ")
    w("$m/(m-1)$ with $m$ the smaller of the distinct true and predicted class counts, that ")
    w("difference multiplies RubricNet's tau-c by roughly 1.021 relative to the forest -- ")
    w("**inflating the score of the model that is worse on the rarest class, precisely ")
    w("because it declines to predict it.**\n")
    w("\n| | scipy tau-c | rescaled to common $m=8$ | tau-b |")
    w("|---|---:|---:|---:|")
    tc_r = kendall_tau_c(yt_p, yp_p_r)
    tc_f = kendall_tau_c(yt_p, yp_p_f)
    w(f"| RubricNet | {fmt(tc_r)} | {fmt(tc_r * ((8/7)/(m_r/(m_r-1))))} | {fmt(kendall_tau_b(yt_p, yp_p_r))} |")
    w(f"| Random Forest | {fmt(tc_f)} | {fmt(tc_f * ((8/7)/(m_f/(m_f-1))))} | {fmt(kendall_tau_b(yt_p, yp_p_f))} |")
    w("\nRoughly half the apparent tau-c gap is normalisation, not model quality. The ")
    w("P(RubricNet > RF) figure for tau-c in the table above is therefore **not** ")
    w("evidence of superiority and should not be quoted. Use tau-b, which does not ")
    w("depend on $m$, or macro-MAE, which penalises the class-7 failure directly.\n")

    # ---- per class
    w("\n## 6. Per-class behaviour (RubricNet)\n")
    _, pc_mae = macro_mae(yt_r, yp_r)
    w("| Class | Grades | N | Recall | MAE |")
    w("|---|---|---:|---:|---:|")
    for c in sorted(set(yt_r.tolist())):
        mask = yt_r == c
        recall = float((yp_r[mask] == c).mean())
        w(f"| {c} | {LEVEL_RANGES[c]} | {int(mask.sum())} | {recall:.3f} | {pc_mae[c]:.3f} |")
    w("\nThe smallest class is estimated from roughly 6 pieces per test fold, so its ")
    w("per-class recall carries a confidence interval far wider than any difference ")
    w("between the models being compared. Per-class numbers are diagnostic, not decisive.\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true", help="reuse cached predictions")
    args = parser.parse_args()

    if args.report_only and os.path.exists(PRED_CACHE):
        with open(PRED_CACHE) as f:
            records = json.load(f)
        print(f"Loaded cached predictions from {PRED_CACHE}")
    else:
        print("Recovering predictions from checkpoints...")
        records = recover_predictions()

    report = build_report(records)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nWrote {REPORT_PATH}")
    print(report)


if __name__ == "__main__":
    main()
