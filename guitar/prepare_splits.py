"""
Phase 1 data/eval scaffolding for the guitar RubricNet experiments.

Produces `guitar/guitar_splits.json`, a fixed 5-fold stratified split (format
mirrors `rubricnet/cipi_splits.json`) so baselines, RubricNet, and the
ablation/feature-selection experiments all train and evaluate on identical data.

Difficulty binning: levels are grouped into 8 classes via equal-frequency
(quantile-style) binning over contiguous level ranges, computed by walking
levels 1-20 in order and closing each bin once it accumulates ~1/8 of the
pieces. Recomputed against the final 716-piece dataset (after merging
newly_matched_pieces.xlsx, dropping 7 confirmed cross-batch duplicates, and
dropping 1 mismatched-content PDF -- see scripts/analysis/find_duplicate_pieces.py
and drop_cross_batch_duplicates.py): [1-3, 4-5, 6-7, 8, 9-10, 11-12, 13-15,
16-20], class sizes 37-136 (3.7x ratio) -- still the best-balanced k among
7/8/9, edges unchanged since the 724-piece run.
"""
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

FEATURE_GROUPS = {
    "lh": [
        "barre_ratio",
        "avg_chord_stretch",
        "max_chord_stretch",
        "avg_position_shift",
        "fret_change_rate",
    ],
    "rh": [
        "arpeggio_density",
        "avg_string_jump",
        "max_string_jump",
        "special_technique_ratio",
    ],
    "global": [
        "avg_polyphony",
        "total_notes",
        "tempo_bpm",
    ],
}
ALL_FEATURES = FEATURE_GROUPS["lh"] + FEATURE_GROUPS["rh"] + FEATURE_GROUPS["global"]

FEATURE_GROUPS_V2 = {
    "lh": [
        "barre_ratio",
        "avg_chord_stretch",
        "max_chord_stretch",
        "avg_position_shift",
        "fret_change_rate",
        "avg_fret",
        "p90_fret",
        "high_position_ratio",
        "open_string_ratio",
        "p90_chord_stretch",
        "shift_rate",
        "max_position_shift",
        "std_position_shift",
        "fret_entropy",
    ],
    "rh": [
        "arpeggio_density",
        "avg_string_jump",
        "max_string_jump",
        "string_entropy",
        "chord_ratio",
    ],
    "global": [
        "avg_polyphony",
        "total_notes",
        "tempo_bpm",
        "log_total_notes",
        "repetition_ratio",
    ],
}
ALL_FEATURES_V2 = FEATURE_GROUPS_V2["lh"] + FEATURE_GROUPS_V2["rh"] + FEATURE_GROUPS_V2["global"]

FEATURE_GROUPS_V3 = {
    "lh": FEATURE_GROUPS_V2["lh"] + [
        "max_avg_chord_stretch_window",
        "p95_position_shift_window",
        "avg_stretch_velocity_beats",
        "p90_stretch_velocity_beats",
        "avg_position_shift_speed_beats",
        "max_position_shift_speed_beats",
    ],
    "rh": FEATURE_GROUPS_V2["rh"] + [
        "polyphonic_arpeggio_intensity_beats",
    ],
    "global": FEATURE_GROUPS_V2["global"] + [
        "max_note_density_window",
    ],
}
ALL_FEATURES_V3 = FEATURE_GROUPS_V3["lh"] + FEATURE_GROUPS_V3["rh"] + FEATURE_GROUPS_V3["global"]

# V3 minus the descriptors feature_audit_v2.md measured at |Spearman rho| <= 0.16
# against raw Difficulty (tempo_bpm, arpeggio_density, fret_change_rate,
# avg_string_jump, chord_ratio, avg_polyphony) but marked "Kept" anyway. In a
# strictly additive model these near-zero-signal descriptors only add noise to
# the summed score, so this set isolates whether dropping them helps.
_DEAD_V3 = {
    "tempo_bpm", "arpeggio_density", "fret_change_rate",
    "avg_string_jump", "chord_ratio", "avg_polyphony",
}
FEATURE_GROUPS_V3_PRUNED = {
    group: [f for f in feats if f not in _DEAD_V3]
    for group, feats in FEATURE_GROUPS_V3.items()
}
ALL_FEATURES_V3_PRUNED = (
    FEATURE_GROUPS_V3_PRUNED["lh"] + FEATURE_GROUPS_V3_PRUNED["rh"] + FEATURE_GROUPS_V3_PRUNED["global"]
)

NUM_CLASSES = 8

# Bin edges from equal-frequency binning over the 724-piece dataset (inclusive level ranges).
_BIN_EDGES = [(1, 3), (4, 5), (6, 7), (8, 8), (9, 10), (11, 12), (13, 15), (16, 20)]


def bin_difficulty(level: int) -> int:
    level = int(level)
    for class_idx, (lo, hi) in enumerate(_BIN_EDGES):
        if lo <= level <= hi:
            return class_idx
    raise ValueError(f"difficulty level {level} out of expected range 1-20")


def make_piece_id(row) -> str:
    return f"{row['Title']}||{row['Composer']}"


def main(csv_path="features/guitar_descriptors.csv", out_path="guitar/guitar_splits.json", n_splits=5, seed=42):
    df = pd.read_csv(csv_path)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    assert df["piece_id"].is_unique, "Title+Composer is not a unique key, need a different piece id"

    df["label"] = df["Difficulty"].apply(bin_difficulty)

    ids = df["piece_id"].values
    labels = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = {}
    for split_idx, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
        train_val_ids, train_val_labels = ids[train_val_idx], labels[train_val_idx]
        test_ids, test_labels = ids[test_idx], labels[test_idx]

        train_idx, val_idx = train_test_split(
            range(len(train_val_ids)), test_size=0.1, stratify=train_val_labels, random_state=seed
        )
        train_ids, train_labels = train_val_ids[train_idx], train_val_labels[train_idx]
        val_ids, val_labels = train_val_ids[val_idx], train_val_labels[val_idx]

        splits[str(split_idx)] = {
            "train": {i: int(l) for i, l in zip(train_ids, train_labels)},
            "val": {i: int(l) for i, l in zip(val_ids, val_labels)},
            "test": {i: int(l) for i, l in zip(test_ids, test_labels)},
        }

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote {out_path}")
    for split_idx, split in splits.items():
        print(
            f"  split {split_idx}: train={len(split['train'])} "
            f"val={len(split['val'])} test={len(split['test'])}"
        )


if __name__ == "__main__":
    main()
