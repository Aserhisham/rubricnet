"""Stability of RubricNet's descriptor-influence ranking across seeds and folds.

Why this exists
---------------
The interpretability analysis in the thesis -- the influence ranking, the monotonicity
plot and the worked rubric example -- is computed on a single checkpoint, fold 0 of
seed 0. Since the sign of the aggregated score is a symmetry the optimiser breaks
independently on each run, and since fold-to-fold spread on the headline metrics is
around 0.04 accuracy, "these descriptors carry the predictions" is a claim that should
be checked across runs rather than read off one of them.

This recomputes the influence range

    Influence(i) = max_x g_i(x_i) - min_x g_i(x_i)

for every one of the 15 checkpoints, and reports how far the rankings agree. Influence
is a magnitude, so it is unaffected by the sign symmetry and needs no normalisation.
"""

import argparse
import itertools
import json
import sys
from statistics import mean, stdev

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR2, NUM_CLASSES
from guitar.train_guitar_rubricnet import Args, load_hyperparams
from rubricnet.rubricnet import RubricnetSklearn


def influence_for(features, splits, columns, alias, seed, split_idx, hyperparams):
    X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
    X_test, y_test = get_fold_xy(features, splits, split_idx, "test")
    medians = X_train.median().fillna(0.0)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    scaler = StandardScaler().fit(X_train)

    clf = RubricnetSklearn(
        input_dim=len(columns), num_classes=NUM_CLASSES, split=split_idx,
        args=Args(alias_experiment=f"{alias}_seed_{seed}", **hyperparams), logging=False,
    )
    clf.load_model(f"checkpoints/{alias}_seed_{seed}/split_{split_idx}.ckpt")
    # predict_descriptor_scores returns one row per descriptor, so the reduction
    # is over axis 1 (the test pieces), not axis 0.
    scores = np.asarray(clf.predict_descriptor_scores(scaler.transform(X_test)))
    assert scores.shape[0] == len(columns), scores.shape
    return dict(zip(columns, scores.max(axis=1) - scores.min(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="guitar_rubricnet_final_v5_pruned_collinear2")
    ap.add_argument("--csv", default="features/guitar_descriptors_v5.csv")
    ap.add_argument("--splits", default="guitar/guitar_splits_v5.json")
    ap.add_argument("--hparams", default="guitar/best_hyperparams_guitar_all_v5.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="guitar/influence_stability.json")
    args = ap.parse_args()

    columns = list(ALL_FEATURES_V5_PRUNED_COLLINEAR2)
    features, splits = load_data(csv_path=args.csv, splits_path=args.splits, columns=columns)
    hyperparams = load_hyperparams(args.hparams)

    runs = {}
    for seed in (int(s) for s in args.seeds.split(",")):
        for split_idx in range(5):
            runs[(seed, split_idx)] = influence_for(
                features, splits, columns, args.alias, seed, split_idx, hyperparams)
    print(f"loaded {len(runs)} checkpoints\n")

    table = pd.DataFrame(runs).T[columns]
    order = table.mean().sort_values(ascending=False)

    print(f"{'descriptor':<38}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'rank sd':>9}")
    print("-" * 79)
    ranks = table.rank(axis=1, ascending=False)
    for feat in order.index:
        col = table[feat]
        print(f"{feat:<38}{col.mean():>8.3f}{col.std():>8.3f}{col.min():>8.3f}"
              f"{col.max():>8.3f}{ranks[feat].std():>9.2f}")

    # pairwise agreement between the 15 rankings
    taus, rhos = [], []
    for a, b in itertools.combinations(runs, 2):
        va = [runs[a][f] for f in columns]
        vb = [runs[b][f] for f in columns]
        taus.append(kendalltau(va, vb).statistic)
        rhos.append(spearmanr(va, vb).statistic)

    topsets = [set(sorted(r, key=r.get, reverse=True)[:args.top]) for r in runs.values()]
    from collections import Counter
    counts = Counter(f for s in topsets for f in s)
    consensus = set(order.index[:args.top])

    print(f"\nPairwise agreement between the {len(runs)} rankings "
          f"({len(taus)} pairs):")
    print(f"  Kendall tau-b   mean {mean(taus):.3f} +/- {stdev(taus):.3f}   min {min(taus):.3f}")
    print(f"  Spearman rho    mean {mean(rhos):.3f} +/- {stdev(rhos):.3f}   min {min(rhos):.3f}")

    print(f"\nHow often each descriptor lands in a run's top {args.top}:")
    for feat, c in counts.most_common(8):
        mark = " (in the pooled top 5)" if feat in consensus else ""
        print(f"  {c:2d}/{len(runs)}  {feat}{mark}")

    stable = [f for f in consensus if counts[f] == len(runs)]
    print(f"\n{len(stable)} of the pooled top {args.top} appear in every single run: "
          f"{', '.join(sorted(stable))}")

    json.dump({
        "n_runs": len(runs),
        "mean_influence": {f: float(order[f]) for f in order.index},
        "std_influence": {f: float(table[f].std()) for f in columns},
        "rank_std": {f: float(ranks[f].std()) for f in columns},
        "pairwise_kendall_tau": {"mean": mean(taus), "std": stdev(taus), "min": min(taus)},
        "pairwise_spearman": {"mean": mean(rhos), "std": stdev(rhos), "min": min(rhos)},
        "top_k": args.top,
        "top_k_appearance_counts": dict(counts),
        "always_top_k": sorted(stable),
    }, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
