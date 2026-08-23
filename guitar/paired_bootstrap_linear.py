"""Paired bootstrap of RubricNet against the linear ordinal baseline.

Why this exists
---------------
guitar/paired_bootstrap_macro_tuned.py compares RubricNet against a Random Forest,
which the unified comparison (guitar/unified_table.py) shows is not the strongest
baseline on this task: an ordinal logistic regression over the same descriptors leads
every tree-based family under every selection criterion tried. This script repoints the
same procedure at that opponent, so the statistical comparison is against the model the
thesis actually has to beat.

Both models are scored on identical resamples of the same test pieces, which removes the
piece-to-piece variance they share. Reported p is the fraction of resamples in which
RubricNet is better, with Holm--Bonferroni correction across the metric family.
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")

from guitar.unified_table import METRICS, score_all

HIGHER_IS_BETTER = {name: direction == "max" for name, _, direction in METRICS}


def load_rubricnet(path, seed):
    d = json.load(open(path))
    runs = [r for r in d["predictions"] if int(r.get("seed", 0)) == seed]
    if not runs:
        raise SystemExit(f"no predictions for seed {seed} in {path}")
    return runs


def load_baseline(path, key):
    d = json.load(open(path))
    if key not in d:
        raise SystemExit(f"{key} not in {path}; have {sorted(k for k in d if isinstance(d[k], dict))}")
    return d[key]["predictions"]


def pool(runs):
    """Concatenate per-fold predictions into one aligned pair of vectors."""
    by_fold = {int(r["fold"]): r for r in runs}
    y_true, y_pred = [], []
    for f in sorted(by_fold):
        y_true.extend(by_fold[f]["y_true"])
        y_pred.extend(by_fold[f]["y_pred"])
    return np.asarray(y_true), np.asarray(y_pred)


def holm(pvals):
    """Holm--Bonferroni over the family; returns the reject decision per entry."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= 0.05 / (m - rank):
            out[i] = True
        else:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubricnet", default="guitar/rubricnet_results_v5_pruned_collinear2.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", default="guitar/unified_baselines.json")
    ap.add_argument("--opponent", default="ordinal_logistic")
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument("--out", default="guitar/paired_bootstrap_vs_ordinal_logistic.json")
    args = ap.parse_args()

    ra_true, ra_pred = pool(load_rubricnet(args.rubricnet, args.seed))
    ba_true, ba_pred = pool(load_baseline(args.baselines, args.opponent))

    if not np.array_equal(ra_true, ba_true):
        raise SystemExit("test pieces are not aligned between the two models; "
                         "the pairing would be meaningless")
    n = len(ra_true)
    print(f"paired over {n} test pieces, {args.resamples} resamples\n")

    rng = np.random.default_rng(0)
    wins = defaultdict(int)
    for _ in range(args.resamples):
        idx = rng.integers(0, n, n)
        a = score_all(ra_true[idx], ra_pred[idx])
        b = score_all(ba_true[idx], ba_pred[idx])
        for name, _, _ in METRICS:
            better = a[name] > b[name] if HIGHER_IS_BETTER[name] else a[name] < b[name]
            wins[name] += int(better)

    point_a = score_all(ra_true, ra_pred)
    point_b = score_all(ba_true, ba_pred)
    rows = [(name, point_a[name], point_b[name], wins[name] / args.resamples)
            for name, _, _ in METRICS]
    rows.sort(key=lambda r: -r[3])
    # p = fraction of resamples where RubricNet is better; small p means the opponent
    # dominates. Pass p directly so holm() sorts ascending with the strongest opponent
    # advantage (smallest p) at rank 1. Using 1-p here was a direction inversion bug.
    survives = holm([r[3] for r in rows])

    print(f"{'metric':<26}{'RubricNet':>12}{'opponent':>12}{'p':>8}  Holm")
    print("-" * 68)
    for (name, a, b, p), s in zip(rows, survives):
        print(f"{name:<26}{a:>12.4f}{b:>12.4f}{p:>8.3f}  {'yes' if s else 'no'}")

    better = sum(1 for _, a, b, p in rows if p > 0.5)
    print(f"\nRubricNet holds the better point estimate on {better} of {len(rows)} measures.")
    print(f"Uncorrected p >= 0.95 on {sum(1 for r in rows if r[3] >= 0.95)}; "
          f"surviving Holm: {sum(survives)}.")

    json.dump({"opponent": args.opponent, "seed": args.seed, "n_pieces": int(n),
               "resamples": args.resamples,
               "results": [{"metric": nm, "rubricnet": a, "opponent": b, "p": p,
                            "survives_holm": bool(s)}
                           for (nm, a, b, p), s in zip(rows, survives)]},
              open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
