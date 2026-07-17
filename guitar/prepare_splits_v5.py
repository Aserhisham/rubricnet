"""V5 dataset: drop the 76 source=='pdf' pieces with has_rhythm==False (dummy
uniform-quarter-note placeholders -- see guitar/per_source_diagnostics.py /
plot_rhythm_diagnostic.py, which showed RubricNet accuracy nearly halves on
these pieces, .329 -> .171). Keeps the other 640 pieces (all of dada_gp/gaps
already have real rhythm; only pdf had the dummy-placeholder gap).

Same protocol as guitar/prepare_splits.py: frozen bin_difficulty edges (not
recomputed -- the class *definition* must stay identical to v1-v4 for results
to be comparable), stratified 5-fold + 90/10 train/val split, same seed=42.

Outputs:
- features/guitar_descriptors_v5.csv (640 rows, same columns as v3)
- guitar/guitar_splits_v5.json (new split file -- piece set differs from
  guitar_splits.json so it cannot reuse the frozen v1-v4 splits)
"""
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from guitar.prepare_splits import bin_difficulty, make_piece_id

N_SPLITS = 5
SEED = 42


def main():
    df = pd.read_csv("features/guitar_descriptors_v3.csv")
    before = len(df)
    dropped = df[(df["source"] == "pdf") & (~df["has_rhythm"])]
    df = df[~((df["source"] == "pdf") & (~df["has_rhythm"]))].reset_index(drop=True)
    print(f"Dropped {len(dropped)} pdf/no-rhythm pieces: {before} -> {len(df)}")

    df.to_csv("features/guitar_descriptors_v5.csv", index=False)
    print("Wrote features/guitar_descriptors_v5.csv")

    df["piece_id"] = df.apply(make_piece_id, axis=1)
    assert df["piece_id"].is_unique
    df["label"] = df["Difficulty"].apply(bin_difficulty)

    ids = df["piece_id"].values
    labels = df["label"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = {}
    for split_idx, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
        train_val_ids, train_val_labels = ids[train_val_idx], labels[train_val_idx]
        test_ids, test_labels = ids[test_idx], labels[test_idx]

        train_idx, val_idx = train_test_split(
            range(len(train_val_ids)), test_size=0.1, stratify=train_val_labels, random_state=SEED
        )
        train_ids, train_labels = train_val_ids[train_idx], train_val_labels[train_idx]
        val_ids, val_labels = train_val_ids[val_idx], train_val_labels[val_idx]

        splits[str(split_idx)] = {
            "train": {i: int(l) for i, l in zip(train_ids, train_labels)},
            "val": {i: int(l) for i, l in zip(val_ids, val_labels)},
            "test": {i: int(l) for i, l in zip(test_ids, test_labels)},
        }

    out_path = "guitar/guitar_splits_v5.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote {out_path}")
    for split_idx, split in splits.items():
        print(f"  split {split_idx}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
