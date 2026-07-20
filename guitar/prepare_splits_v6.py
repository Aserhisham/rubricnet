"""V6 dataset: paper-comparison labeling and split protocol.

Same 640 pieces and descriptors as V5 (features/guitar_descriptors_v5.csv is
reused as-is; no new CSV), but re-labeled and re-split to mirror the original
RubricNet paper's CIPI setup (Ramoneda et al. 2024) as closely as the
GuitarBurst 1-20 scale allows:

- 9 classes, equal-width over levels 1-20 (Henle's 9 grades are equal-width
  editorial units; CIPI never bins). Widths are as even as 20/9 permits:
  [1-2, 3-4, 5-6, 7-8, 9-11, 12-13, 14-15, 16-17, 18-20].
- Split protocol from the paper: 5-fold stratified CV where each split is
  60% train / 20% val / 20% test (the fold is the 20% test; the remaining
  80% is split 75/25 into train/val), vs. v1-v5's 72/8/20.

Note this labeling is intentionally NOT comparable to v1-v5 results (different
class count and boundaries); it exists only for the cross-paper comparison.
Expected from the sigma=3.2 error simulation: ~0.30 exact accuracy, i.e. the
equal-width geometry is slightly harsher than v5's equal-frequency bins for
this label distribution (imbalance ~12.5x vs 4x).

Outputs: guitar/guitar_splits_v6.json
"""
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from guitar.prepare_splits import make_piece_id

N_SPLITS = 5
SEED = 42
NUM_CLASSES_V6 = 9

# Equal-width partition of 1-20 into 9 contiguous grades (widths 2,2,2,2,3,2,2,2,3).
BIN_EDGES_V6 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 11), (12, 13), (14, 15), (16, 17), (18, 20)]


def bin_difficulty_v6(level: int) -> int:
    level = int(level)
    for class_idx, (lo, hi) in enumerate(BIN_EDGES_V6):
        if lo <= level <= hi:
            return class_idx
    raise ValueError(f"difficulty level {level} out of expected range 1-20")


def main():
    df = pd.read_csv("features/guitar_descriptors_v5.csv")
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    assert df["piece_id"].is_unique
    df["label"] = df["Difficulty"].apply(bin_difficulty_v6)

    counts = df["label"].value_counts().sort_index()
    print(f"{len(df)} pieces, {NUM_CLASSES_V6} classes: {counts.to_dict()}")
    print(f"imbalance {counts.max()}/{counts.min()} = {counts.max() / counts.min():.1f}x")

    ids = df["piece_id"].values
    labels = df["label"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = {}
    for split_idx, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
        train_val_ids, train_val_labels = ids[train_val_idx], labels[train_val_idx]
        test_ids, test_labels = ids[test_idx], labels[test_idx]

        # Paper protocol: 60/20/20 overall -> val is 25% of the 80% train+val pool.
        train_idx, val_idx = train_test_split(
            range(len(train_val_ids)), test_size=0.25, stratify=train_val_labels, random_state=SEED
        )
        train_ids, train_labels = train_val_ids[train_idx], train_val_labels[train_idx]
        val_ids, val_labels = train_val_ids[val_idx], train_val_labels[val_idx]

        splits[str(split_idx)] = {
            "train": {i: int(l) for i, l in zip(train_ids, train_labels)},
            "val": {i: int(l) for i, l in zip(val_ids, val_labels)},
            "test": {i: int(l) for i, l in zip(test_ids, test_labels)},
        }

    out_path = "guitar/guitar_splits_v6.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote {out_path}")
    for split_idx, split in splits.items():
        print(f"  split {split_idx}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
