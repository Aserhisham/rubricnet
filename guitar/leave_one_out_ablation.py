"""Per-descriptor leave-one-out ablation: what does each of the 25 descriptors contribute?

Why
---
The thesis justifies its descriptor set collectively but not individually. Section
`sec:pruning` shows that removing the *six weakest descriptors at once* hurt accuracy
(0.310 -> 0.302), and that result is used to justify retaining all of them. But it says
nothing about any single descriptor, which leaves several with no defense at all:

    chord_ratio        rho -0.031   no literature analogue, no expert endorsement
    avg_string_jump    rho -0.098   no literature analogue, no expert endorsement
    arpeggio_density   rho -0.132   retained as the context-free parent of a kept descriptor
    tempo_bpm          rho -0.110   expert-named, but weak and missing for 14 pieces

At a defense, "what does chord_ratio contribute?" currently has no answer. This script
produces one: drop each descriptor individually, retrain, and measure the cost.

Two models, deliberately
------------------------
* **Random Forest** over all 25, matching the per-dimension ablation in
  `AIM-thesis/chapters/05-evaluation.tex` (Table `tab:dimension_ablation`), which uses the
  forest specifically "to isolate the feature groups from RubricNet's training variance".
  Cheap enough to run for every descriptor, and the results are directly comparable to the
  ablation already in the thesis.
* **RubricNet** for a named subset, because the forest can route around a redundant input
  while an additive model must score and sum it. A descriptor can therefore be worthless to
  the forest and still matter to RubricNet, or vice versa; the architecture the thesis
  actually ships is the one whose answer counts.

Reading the output
------------------
A *positive* delta means the metric got worse when the descriptor was removed, i.e. the
descriptor was earning its place. Deltas should be read against the fold-to-fold standard
deviation printed alongside: single-descriptor effects on 640 pieces are small, and most
will land inside noise. That is itself the finding -- it distinguishes "this descriptor is
load-bearing" from "this descriptor is harmless but inert", which are different defenses
and should not be conflated.

Usage
-----
    python -m guitar.leave_one_out_ablation --model rf
    python -m guitar.leave_one_out_ablation --model rubricnet --only chord_ratio,avg_string_jump
"""

import argparse
import hashlib
import json
import os
import sys
from statistics import mean, stdev

import numpy as np
import pandas as pd

os.environ.setdefault("WANDB_MODE", "disabled")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from guitar.baselines import get_fold_xy, load_data
from guitar.metric_study import all_metrics
from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR2, NUM_CLASSES, make_piece_id
from guitar.train_guitar_rubricnet import DEFAULT_HYPERPARAMS
from guitar.tuned_baselines import INNER_FOLDS, SEARCH_SPACES, SEED

CSV_PATH = "features/guitar_descriptors_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
BEST_HYPERPARAMS_PATH = "guitar/best_hyperparams_guitar_all_v5.json"
OUT_JSON = "guitar/leave_one_out_results.json"
REPORT_PATH = "guitar/DESCRIPTOR_JUSTIFICATION.md"

REPORT_METRICS = ["accuracy", "balanced_accuracy", "mae", "kendall_tau_b"]
# Metrics where a larger value is better; used to sign the deltas consistently so that
# "positive = removing it hurt" holds for every column.
HIGHER_BETTER = {"accuracy": True, "balanced_accuracy": True, "mae": False, "kendall_tau_b": True}


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _columns_digest(columns):
    """Stable short digest of a feature set, used to keep checkpoint aliases distinct.

    Every ablation trains a differently-shaped model, so they cannot share a checkpoint
    directory: a 24-descriptor model loading the 25-descriptor baseline's checkpoint fails
    on the extra descriptor_layers entry. Python's built-in hash() is salted per process
    and would break resumability across runs, so this uses a stable digest instead.
    """
    return hashlib.md5("|".join(columns).encode()).hexdigest()[:8]


def load_frames():
    df = pd.read_csv(CSV_PATH)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    df = df.set_index("piece_id")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    return df, splits


def fold_pool(df, splits, columns, split_idx):
    X_tr, y_tr = get_fold_xy(df[columns], splits, split_idx, "train")
    X_va, y_va = get_fold_xy(df[columns], splits, split_idx, "val")
    X_te, y_te = get_fold_xy(df[columns], splits, split_idx, "test")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# --------------------------------------------------------------------------------
def run_rf(df, splits, columns, n_iter):
    estimator, space = SEARCH_SPACES["random_forest"]
    fold_metrics = []
    for split_idx in range(5):
        X_tr, y_tr, X_va, y_va, X_te, y_te = fold_pool(df, splits, columns, split_idx)
        X_pool = pd.concat([X_tr, X_va])
        y_pool = pd.concat([y_tr, y_va])
        med = X_pool.median().fillna(0.0)
        X_pool, X_te = X_pool.fillna(med), X_te.fillna(med)

        search = RandomizedSearchCV(
            estimator=estimator, param_distributions=space, n_iter=n_iter,
            scoring="balanced_accuracy",
            cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED),
            random_state=SEED + split_idx, n_jobs=-1, refit=True,
        )
        search.fit(X_pool, y_pool)
        fold_metrics.append(all_metrics(y_te.to_numpy(), search.best_estimator_.predict(X_te)))
    return fold_metrics


def run_rubricnet(df, splits, columns, seeds=(0, 1, 2)):
    import lightning.pytorch as pl
    from rubricnet.rubricnet import RubricnetSklearn

    with open(BEST_HYPERPARAMS_PATH) as f:
        hp = dict(DEFAULT_HYPERPARAMS)
        hp.update(json.load(f)["params"])

    fold_metrics = []
    for seed in seeds:
        pl.seed_everything(seed, verbose=False)
        for split_idx in range(5):
            X_tr, y_tr, X_va, y_va, X_te, y_te = fold_pool(df, splits, columns, split_idx)
            med = X_tr.median().fillna(0.0)
            X_tr, X_va, X_te = X_tr.fillna(med), X_va.fillna(med), X_te.fillna(med)
            sc = StandardScaler().fit(X_tr)

            # The alias must be unique per feature set. Sharing one alias across the
            # baseline and every ablation means a 24-descriptor model tries to load the
            # 25-descriptor checkpoint left behind by the previous run, which fails on the
            # extra descriptor_layers entry.
            alias = f"loo_tmp_seed{seed}_n{len(columns)}_{_columns_digest(columns)}"
            clf = RubricnetSklearn(
                input_dim=len(columns), num_classes=NUM_CLASSES, split=split_idx,
                args=Args(alias_experiment=alias, **hp), logging=False,
            )
            clf.fit(sc.transform(X_tr), y_tr, sc.transform(X_va), y_va, sc.transform(X_te), y_te)
            clf.load_model(f"checkpoints/{alias}/split_{split_idx}.ckpt")
            y_pred = np.clip(clf.predict(sc.transform(X_te)).cpu().numpy(), 0, NUM_CLASSES - 1)
            fold_metrics.append(all_metrics(y_te.to_numpy(), y_pred))
    return fold_metrics


# --------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["rf", "rubricnet"], default="rf")
    p.add_argument("--n-iter", type=int, default=25, help="RF search budget per fold")
    p.add_argument("--only", default="", help="comma-separated descriptors to ablate (default: all 25)")
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()

    df, splits = load_frames()
    full = [c for c in ALL_FEATURES_V5_PRUNED_COLLINEAR2 if c in df.columns]
    targets = [t.strip() for t in args.only.split(",") if t.strip()] or list(full)
    seeds = tuple(int(s) for s in args.seeds.split(","))

    runner = (lambda cols: run_rf(df, splits, cols, args.n_iter)) if args.model == "rf" \
        else (lambda cols: run_rubricnet(df, splits, cols, seeds))

    print(f"Model    : {args.model}")
    print(f"Baseline : all {len(full)} descriptors")
    print(f"Ablating : {len(targets)} descriptors\n", flush=True)

    print("--- baseline (all descriptors) ---", flush=True)
    base = runner(full)
    base_mean = {m: mean([x[m] for x in base]) for m in REPORT_METRICS}
    base_std = {m: stdev([x[m] for x in base]) for m in REPORT_METRICS}
    print("  " + "  ".join(f"{m}={base_mean[m]:.4f}±{base_std[m]:.3f}" for m in REPORT_METRICS), flush=True)

    results = {"model": args.model, "baseline": {m: [x[m] for x in base] for m in REPORT_METRICS}, "ablations": {}}

    for i, feat in enumerate(targets, 1):
        cols = [c for c in full if c != feat]
        fm = runner(cols)
        results["ablations"][feat] = {m: [x[m] for x in fm] for m in REPORT_METRICS}
        deltas = {}
        for m in REPORT_METRICS:
            d = mean([x[m] for x in fm]) - base_mean[m]
            # sign so that positive always means "removing it made things worse"
            deltas[m] = -d if HIGHER_BETTER[m] else d
        print(f"[{i}/{len(targets)}] drop {feat:34s} " +
              "  ".join(f"d{m[:4]}={deltas[m]:+.4f}" for m in REPORT_METRICS), flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ---- report
    rows = []
    for feat, fm in results["ablations"].items():
        d = {}
        for m in REPORT_METRICS:
            raw = mean(fm[m]) - base_mean[m]
            d[m] = -raw if HIGHER_BETTER[m] else raw
        rows.append((d["accuracy"], feat, d))
    rows.sort(reverse=True)

    L = [f"# Per-Descriptor Justification: Leave-One-Out Ablation ({args.model})\n"]
    L.append("Each row drops exactly one descriptor from the 25 and retrains. ")
    L.append("**Positive = removing it made the model worse**, i.e. the descriptor earns its place.\n")
    L.append(f"\nBaseline (all {len(full)}): " +
             ", ".join(f"{m} {base_mean[m]:.4f} ± {base_std[m]:.3f}" for m in REPORT_METRICS) + "\n")
    L.append("\n| Descriptor dropped | ΔAccuracy | ΔBal.Acc | ΔMAE | Δτ-b | Beyond fold noise? |")
    L.append("|---|---:|---:|---:|---:|:---:|")
    for _, feat, d in rows:
        beyond = "**yes**" if abs(d["accuracy"]) > base_std["accuracy"] else "no"
        L.append(f"| `{feat}` | {d['accuracy']:+.4f} | {d['balanced_accuracy']:+.4f} | "
                 f"{d['mae']:+.4f} | {d['kendall_tau_b']:+.4f} | {beyond} |")
    L.append(f"\nFold-to-fold std of the baseline is ±{base_std['accuracy']:.3f} accuracy, so any ")
    L.append("single-descriptor delta smaller than that is inert rather than load-bearing. ")
    L.append("A descriptor being inert is not an argument for removing it -- the blanket-pruning ")
    L.append("experiment (thesis §Pruning Weak Descriptors) showed removing six inert descriptors ")
    L.append("at once *hurt* -- but it does mean its defense is 'harmless and mildly regularising', ")
    L.append("not 'individually predictive', and it should be described that way.\n")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L))
    print(f"\nWrote {OUT_JSON} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
