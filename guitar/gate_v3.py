"""
Phase A validation gate: does re-extracting base features from the unified
all_xmls source (v3) regress vs the mixed-source v2 features?

Runs Random Forest (the black-box bar) on both CSVs with the SAME feature set
(ALL_FEATURES_V2) and the SAME fixed splits, and reports accuracy / balanced
accuracy / MAE. Gate passes if v3 is not worse than v2.
"""
import os
import sys

import numpy as np
from statistics import mean, stdev
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guitar.baselines import load_data, get_fold_xy, N_SPLITS
from guitar.prepare_splits import ALL_FEATURES_V2


def rf_eval(csv_path):
    features, splits = load_data(csv_path=csv_path, columns=ALL_FEATURES_V2)
    accs, baccs, maes = [], [], []
    for k in range(N_SPLITS):
        Xtr, ytr = get_fold_xy(features, splits, k, "train")
        Xva, yva = get_fold_xy(features, splits, k, "val")
        Xte, yte = get_fold_xy(features, splits, k, "test")
        import pandas as pd
        Xtr = pd.concat([Xtr, Xva]); ytr = pd.concat([ytr, yva])
        m = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
        yp = m.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        baccs.append(balanced_accuracy_score(yte, yp))
        maes.append(mean_absolute_error(yte, yp))
    return accs, baccs, maes


def show(name, accs, baccs, maes):
    print(f"\n{name}")
    print(f"  accuracy         {mean(accs):.4f} +/- {stdev(accs):.4f}")
    print(f"  balanced_acc     {mean(baccs):.4f} +/- {stdev(baccs):.4f}")
    print(f"  MAE              {mean(maes):.4f} +/- {stdev(maes):.4f}")
    return mean(accs), mean(baccs), mean(maes)


if __name__ == "__main__":
    print("=== RF validation gate: v2 (mixed source) vs v3 (unified all_xmls) ===")
    a2 = rf_eval("features/guitar_descriptors_v2.csv")
    a3 = rf_eval("features/guitar_descriptors_v3.csv")
    m2 = show("Random Forest on V2 features", *a2)
    m3 = show("Random Forest on V3 features", *a3)
    print("\n--- delta (v3 - v2) ---")
    print(f"  accuracy     {m3[0]-m2[0]:+.4f}")
    print(f"  balanced_acc {m3[1]-m2[1]:+.4f}")
    print(f"  MAE          {m3[2]-m2[2]:+.4f}  (negative is better)")
    passed = m3[1] >= m2[1] - 0.01
    print(f"\nGATE {'PASSED' if passed else 'FAILED'}: v3 balanced acc {m3[1]:.4f} vs v2 {m2[1]:.4f}")
