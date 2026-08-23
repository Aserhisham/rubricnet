"""Re-decode the finished V5-pruned-collinear2 checkpoints under alternative rules.

Pure inference on the 15 checkpoints already saved under
checkpoints/guitar_rubricnet_final_v5_pruned_collinear2_seed_{0,1,2}/ -- no retraining,
so the model being scored is exactly the one every headline number in the thesis
reports.

The question is whether "class 7 is never predicted" is a property of the data or of
the decoding rule. RubricNet decodes by counting leading cumulative outputs above 0.5
(`(p > 0.5).cumprod(1).sum(1) - 1`, then clipped to [0, K-1]), which requires ALL K
outputs to clear the threshold simultaneously before the top class can ever be emitted,
and silently floors an underflow to class 0. Two standard alternatives that do not have
that asymmetry are scored here on the identical outputs:

  threshold  the shipped rule (reproduces the thesis numbers)
  argmax     P(y = j) = p_j - p_{j+1}, clipped at 0 and renormalised, then argmax
             -- the rule the proportional-odds baseline uses
  expected   E[y] = sum_{j>=1} P(y >= j) = sum_{j=1}^{K-1} p_j, rounded

Outputs guitar/decoding_rule_audit.json.
"""
import json
import os
import sys
from statistics import mean, stdev

import numpy as np
import torch

os.environ.setdefault("WANDB_MODE", "disabled")
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR2, NUM_CLASSES
from guitar.train_guitar_rubricnet import DEFAULT_HYPERPARAMS, Args, compute_metrics
from rubricnet.rubricnet import RubricnetSklearn

CSV_PATH = "features/guitar_descriptors_v5.csv"
SPLITS_PATH = "guitar/guitar_splits_v5.json"
ALIAS = "guitar_rubricnet_final_v5_pruned_collinear2"
BEST_HYPERPARAMS_PATH = "guitar/best_hyperparams_guitar_all_v5.json"
OUT_PATH = "guitar/decoding_rule_audit.json"
CKPT_DIR = "checkpoints"


def load_hyperparams(path):
    with open(path) as f:
        tuned = json.load(f)["params"]
    hp = dict(DEFAULT_HYPERPARAMS)
    hp.update(tuned)
    return hp


def decode_threshold(p):
    """The shipped rule: count leading cumulative outputs above 0.5."""
    return np.clip((p > 0.5).cumprod(axis=1).sum(axis=1) - 1, 0, NUM_CLASSES - 1)


def _class_probs(p):
    """P(y = j) from the cumulative outputs P(y >= j), clipped and renormalised."""
    nxt = np.concatenate([p[:, 1:], np.zeros((p.shape[0], 1))], axis=1)
    probs = np.clip(p - nxt, 0.0, None)
    total = probs.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return probs / total


def decode_argmax(p):
    return _class_probs(p).argmax(axis=1)


def decode_expected(p):
    return np.clip(np.rint(p[:, 1:].sum(axis=1)), 0, NUM_CLASSES - 1).astype(int)


RULES = {"threshold": decode_threshold, "argmax": decode_argmax, "expected": decode_expected}


def main():
    features, splits = load_data(csv_path=CSV_PATH, splits_path=SPLITS_PATH,
                                 columns=ALL_FEATURES_V5_PRUNED_COLLINEAR2)
    hyperparams = load_hyperparams(BEST_HYPERPARAMS_PATH)

    per_rule = {r: {"accuracy": [], "balanced_accuracy": [], "mae": [], "kendall_tau": [],
                    "acc_plus_minus_1": [], "n_pred_class7": 0, "n_pred_class0": 0}
                for r in RULES}
    n_true7 = 0
    n_total = 0

    for seed in (0, 1, 2):
        alias = f"{ALIAS}_seed_{seed}"
        for split_idx in range(5):
            X_train, _ = get_fold_xy(features, splits, split_idx, "train")
            X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

            medians = X_train.median().fillna(0.0)
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)
            scaler = StandardScaler().fit(X_train)

            clf = RubricnetSklearn(input_dim=len(ALL_FEATURES_V5_PRUNED_COLLINEAR2),
                                   num_classes=NUM_CLASSES, split=split_idx,
                                   args=Args(alias_experiment=alias, **hyperparams),
                                   logging=False)
            clf.load_model(f"{CKPT_DIR}/{alias}/split_{split_idx}.ckpt")

            model = clf.model
            model.eval()
            with torch.no_grad():
                xt = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(model.device)
                p = model(xt).cpu().numpy()

            y_true = np.asarray(y_test)
            n_true7 += int((y_true == 7).sum())
            n_total += len(y_true)

            for rule, fn in RULES.items():
                y_pred = np.asarray(fn(p), dtype=int)
                m = compute_metrics(y_true, y_pred)
                for k in ("accuracy", "balanced_accuracy", "mae", "kendall_tau", "acc_plus_minus_1"):
                    per_rule[rule][k].append(m[k])
                per_rule[rule]["n_pred_class7"] += int((y_pred == 7).sum())
                per_rule[rule]["n_pred_class0"] += int((y_pred == 0).sum())

    out = {"n_runs": 15, "n_test_pieces_pooled": n_total, "n_true_class7_pooled": n_true7, "rules": {}}
    print(f"\npooled over 15 runs: {n_total} scored pieces, {n_true7} of them truly class 7\n")
    for rule in RULES:
        r = per_rule[rule]
        entry = {"n_pred_class7": r["n_pred_class7"], "n_pred_class0": r["n_pred_class0"]}
        for k in ("accuracy", "balanced_accuracy", "mae", "kendall_tau", "acc_plus_minus_1"):
            entry[k] = {"mean": float(mean(r[k])), "std": float(stdev(r[k]))}
        out["rules"][rule] = entry
        print(f"{rule:10s} acc={entry['accuracy']['mean']:.4f}+/-{entry['accuracy']['std']:.4f} "
              f"bacc={entry['balanced_accuracy']['mean']:.4f} "
              f"MAE={entry['mae']['mean']:.4f} tau={entry['kendall_tau']['mean']:.4f} "
              f"acc+-1={entry['acc_plus_minus_1']['mean']:.4f} "
              f"| predicted class 7: {entry['n_pred_class7']}  class 0: {entry['n_pred_class0']}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
