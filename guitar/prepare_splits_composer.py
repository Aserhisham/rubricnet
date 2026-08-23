"""Composer-grouped 5-fold splits: no composer appears in both train and test.

Motivation
----------
The frozen splits of `guitar/prepare_splits_v5.py` are stratified at the piece level, which
means a composer's works are scattered across folds. On this corpus that is not a minor
leak: 523 of 640 pieces (81.7%) belong to one of 23 composers holding five or more pieces,
and every one of those 23 spans all five test folds. Bach contributes 85 pieces, Sor 61,
Giuliani 52, Barrios 48. Worse, graded opus sets of near-identical studies (Sor Op. 60,
Carcassi Op. 60, Aguado's lessons) straddle folds, so the model can meet a study's siblings
in training and then be tested on it.

Piece-level cross-validation therefore never asks the question a teacher would: how does
this behave on a composer it has never seen? These splits ask it. They are a robustness
check reported alongside the frozen splits, not a replacement for them -- the headline
numbers stay on the frozen splits so that every generation remains comparable.

Output: guitar/guitar_splits_v5_composer.json, same schema as the piece-level splits.
"""
import json, os, sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from guitar.prepare_splits import make_piece_id

SRC_SPLITS = "guitar/guitar_splits_v5.json"
OUT = "guitar/guitar_splits_v5_composer.json"
SEED = 42


def main():
    src = json.load(open(SRC_SPLITS))
    labels = {}
    for f in map(str, range(5)):
        for sub in ("train", "val", "test"):
            labels.update(src[f][sub])

    ids = sorted(labels)
    y = np.array([labels[i] for i in ids])
    groups = np.array([i.split("||")[1].strip() for i in ids])
    print(f"{len(ids)} pieces, {len(set(groups))} composers")

    out = {}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (trv_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(ids)), y, groups)):
        trv_ids = [ids[i] for i in trv_idx]
        trv_y = y[trv_idx]
        trv_groups = groups[trv_idx]

        # Hold out whole composers for validation too, so val is a fair proxy for test.
        uniq = sorted(set(trv_groups))
        rng = np.random.default_rng(SEED + fold)
        rng.shuffle(uniq)
        val_groups, n = set(), 0
        target = int(round(0.10 * len(trv_ids)))
        for g in uniq:
            if n >= target:
                break
            cnt = int((trv_groups == g).sum())
            # keep val from being dominated by one huge composer
            if cnt > 0.4 * target and n > 0:
                continue
            val_groups.add(g); n += cnt

        tr = {p: int(labels[p]) for p, g in zip(trv_ids, trv_groups) if g not in val_groups}
        va = {p: int(labels[p]) for p, g in zip(trv_ids, trv_groups) if g in val_groups}
        te = {ids[i]: int(y[i]) for i in te_idx}

        # sanity: no composer may appear on both sides of any boundary
        gt = {p.split("||")[1].strip() for p in tr}
        gv = {p.split("||")[1].strip() for p in va}
        ge = {p.split("||")[1].strip() for p in te}
        assert not (gt & ge), f"fold {fold}: composer leak train/test"
        assert not (gv & ge), f"fold {fold}: composer leak val/test"
        assert not (gt & gv), f"fold {fold}: composer leak train/val"

        out[str(fold)] = {"train": tr, "val": va, "test": te}
        print(f"  fold {fold}: train={len(tr):3d} val={len(va):3d} test={len(te):3d} | "
              f"test classes={sorted(Counter(te.values()).items())}")

    json.dump(out, open(OUT, "w"))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
