"""
Phase A gate (RubricNet): quick single-seed 5-fold RubricNet on v2 vs v3 base
features, reusing the tuned v2 hyperparameters. This is the additive-model test
of whether the cleaner unified source helps (RF was flat but can mask features
via interactions; RubricNet cannot).
"""
import json
import os
import sys
from statistics import mean, stdev

import numpy as np
import lightning.pytorch as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V2, NUM_CLASSES
from guitar.train_guitar_rubricnet import Args, DEFAULT_HYPERPARAMS
from rubricnet.rubricnet import RubricnetSklearn

with open("guitar/best_hyperparams_guitar_all_v2.json") as f:
    HP = dict(DEFAULT_HYPERPARAMS); HP.update(json.load(f)["params"])


def run(csv_path, tag, seed=0):
    features, splits = load_data(csv_path=csv_path, columns=ALL_FEATURES_V2)
    pl.seed_everything(seed, workers=True)
    accs, baccs, maes = [], [], []
    for k in range(5):
        Xtr, ytr = get_fold_xy(features, splits, k, "train")
        Xva, yva = get_fold_xy(features, splits, k, "val")
        Xte, yte = get_fold_xy(features, splits, k, "test")
        sc = StandardScaler().fit(Xtr)
        args = Args(alias_experiment=f"gate_{tag}", **HP)
        clf = RubricnetSklearn(input_dim=len(ALL_FEATURES_V2), num_classes=NUM_CLASSES,
                               split=k, args=args, logging=False)
        clf.fit(sc.transform(Xtr), ytr, sc.transform(Xva), yva, sc.transform(Xte), yte)
        clf.load_model(f"checkpoints/gate_{tag}/split_{k}.ckpt")
        yp = clf.predict(sc.transform(Xte)).cpu().numpy()
        accs.append(accuracy_score(yte, yp))
        baccs.append(balanced_accuracy_score(yte, yp))
        maes.append(mean_absolute_error(yte, yp))
        print(f"    [{tag}] split {k}: acc={accs[-1]:.4f} bacc={baccs[-1]:.4f}")
    print(f"  {tag}: acc={mean(accs):.4f}±{stdev(accs):.4f}  "
          f"bacc={mean(baccs):.4f}±{stdev(baccs):.4f}  mae={mean(maes):.4f}")
    return mean(accs), mean(baccs), mean(maes)


if __name__ == "__main__":
    print("RubricNet gate (seed 0, v2 hyperparams):")
    v2 = run("features/guitar_descriptors_v2.csv", "v2")
    v3 = run("features/guitar_descriptors_v3.csv", "v3")
    print(f"\ndelta v3-v2: acc {v3[0]-v2[0]:+.4f}  bacc {v3[1]-v2[1]:+.4f}  mae {v3[2]-v2[2]:+.4f}")
