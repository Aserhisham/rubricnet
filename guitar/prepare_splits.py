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

# Option A: "V3 base" is the V2/unified descriptor set WITHOUT the eight
# rhythm-aware window/velocity features -- i.e. exactly the V2 feature groups,
# but read from the unified-provenance v3 CSV. This isolates whether the rhythm
# features helped at all before layering interaction features on top.
FEATURE_GROUPS_V3_BASE = FEATURE_GROUPS_V2
ALL_FEATURES_V3_BASE = ALL_FEATURES_V2

# Five hand-crafted interaction descriptors (see
# guitar_features.calculate_interaction_descriptors_v3). Kept in a dedicated
# group so importance plots read cleanly.
FEATURE_GROUP_INTERACTION = [
    "barre_difficulty_tempo",
    "stretch_under_time_pressure",
    "position_shift_entropy",
    "open_string_efficiency",
    "arpeggio_stretch_coupling",
]
FEATURE_GROUPS_V3_BASE_NEW = {
    **FEATURE_GROUPS_V3_BASE,
    "interaction": FEATURE_GROUP_INTERACTION,
}
ALL_FEATURES_V3_BASE_NEW = ALL_FEATURES_V3_BASE + FEATURE_GROUP_INTERACTION

# Expert-review pruning (guitar notation domain expert, 2026-07-17, see
# guitar/EXPERT_MEETING.md): avg_polyphony called "shouldn't be important"
# (matches near-zero rho=+0.06). fret_change_rate: expert flagged as
# important in discussion but ruled it out on the feature sheet; an
# events-based denominator reformulation was tested (algebraically
# reconstructed from total_notes/avg_polyphony, no re-extraction) and made
# the correlation MORE negative (-0.143 -> -0.158), not less -- so the
# negative sign isn't a normalization artifact of the original notes-based
# denominator. It's strongly anti-correlated with repetition_ratio
# (rho=-0.68), which IS positively linked to difficulty and already covers
# the same underlying concept with the correct sign. Both dropped.
_EXPERT_DROPPED_V5 = {"avg_polyphony", "fret_change_rate"}
FEATURE_GROUPS_V5_PRUNED = {
    group: [f for f in feats if f not in _EXPERT_DROPPED_V5]
    for group, feats in FEATURE_GROUPS_V3.items()
}
ALL_FEATURES_V5_PRUNED = (
    FEATURE_GROUPS_V5_PRUNED["lh"] + FEATURE_GROUPS_V5_PRUNED["rh"] + FEATURE_GROUPS_V5_PRUNED["global"]
)

# Remaining expert-ruled-out descriptors (see guitar/EXPERT_MEETING.md), split
# into two tiers by risk profile -- unlike avg_polyphony/fret_change_rate,
# high_position_ratio/avg_fret/p90_fret are individually STRONG predictors
# (rho +0.53 to +0.58) ruled out only for being 0.91-0.94 collinear with kept
# fret_entropy. Dropping them could cut redundant double-counted signal (good)
# or remove signal the additive model actually relies on (bad) -- this must be
# decided on VALIDATION accuracy, not test, since test has already been used
# to select several prior feature-set variants this session (see
# guitar/EXPERT_FEATURE_REVIEW.md methodology note).
_COLLINEAR_CLUSTER_V5 = {"high_position_ratio", "avg_fret", "p90_fret"}
FEATURE_GROUPS_V5_PRUNED_COLLINEAR = {
    group: [f for f in feats if f not in _COLLINEAR_CLUSTER_V5]
    for group, feats in FEATURE_GROUPS_V5_PRUNED.items()
}
ALL_FEATURES_V5_PRUNED_COLLINEAR = (
    FEATURE_GROUPS_V5_PRUNED_COLLINEAR["lh"] + FEATURE_GROUPS_V5_PRUNED_COLLINEAR["rh"]
    + FEATURE_GROUPS_V5_PRUNED_COLLINEAR["global"]
)

# Remaining weak/moderate expert-ruled-out descriptors (rho -0.13 to +0.30),
# same risk profile as avg_polyphony/fret_change_rate. Layered on top of the
# collinear-cluster drop.
_REST_EXPERT_DROP_V5 = {"max_string_jump", "arpeggio_density", "avg_string_jump", "chord_ratio", "tempo_bpm"}
FEATURE_GROUPS_V5_PRUNED_FULL = {
    group: [f for f in feats if f not in _REST_EXPERT_DROP_V5]
    for group, feats in FEATURE_GROUPS_V5_PRUNED_COLLINEAR.items()
}
ALL_FEATURES_V5_PRUNED_FULL = (
    FEATURE_GROUPS_V5_PRUNED_FULL["lh"] + FEATURE_GROUPS_V5_PRUNED_FULL["rh"] + FEATURE_GROUPS_V5_PRUNED_FULL["global"]
)

# The 7 new descriptor candidates the expert named but that didn't exist in
# any prior descriptor set (see guitar/EXPERT_MEETING.md "New feature
# candidates surfaced this meeting" and guitar/guitar_features.py
# EXPERT_NEW_COLUMNS / calculate_expert_new_descriptors). Layered on top of
# the adopted 27-feature V5-pruned-collinear set (the current best-tested
# default -- see guitar/EXPERT_FEATURE_REVIEW.md). Requires
# features/guitar_descriptors_v5_expert_new.csv (from
# guitar/backfill_expert_new_descriptors_v5.py), not the plain V5 csv, since
# these columns don't exist there. As with the collinear-cluster/rest-of-list
# decisions above, this must be decided on VALIDATION accuracy, not test.
FEATURE_GROUP_EXPERT_NEW = [
    "onset_rate_bps",
    "max_onset_rate_bps",
    "chord_change_ratio",
    "n_meter_changes",
    "irregular_meter_ratio",
    "note_duration_entropy",
    "finest_subdivision_rank",
]
FEATURE_GROUPS_V5_PRUNED_COLLINEAR_EXPERT_NEW = {
    **FEATURE_GROUPS_V5_PRUNED_COLLINEAR,
    "expert_new": FEATURE_GROUP_EXPERT_NEW,
}
ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW = ALL_FEATURES_V5_PRUNED_COLLINEAR + FEATURE_GROUP_EXPERT_NEW

# Trimmed follow-up: the full 7-descriptor addition above was a wash on
# validation (all 4 metrics moved within ~1 fold-std of the 27-feature
# baseline, 2 up/2 down -- see guitar/EXPERT_FEATURE_REVIEW.md). n_meter_changes
# and irregular_meter_ratio were the weakest of the 7 by BOTH independent
# signals -- lowest |Spearman rho| (+0.06, +0.17) AND lowest Random Forest
# importance (0.0017, 0.0048, an order of magnitude below the other five) --
# so this drops just those two "takte"/meter-complexity descriptors and keeps
# the tempo-onset-rate, chord_change_ratio, and note-duration-entropy trio.
FEATURE_GROUP_EXPERT_NEW_TRIMMED = [
    "onset_rate_bps",
    "max_onset_rate_bps",
    "chord_change_ratio",
    "note_duration_entropy",
    "finest_subdivision_rank",
]
FEATURE_GROUPS_V5_PRUNED_COLLINEAR_EXPERT_NEW_TRIMMED = {
    **FEATURE_GROUPS_V5_PRUNED_COLLINEAR,
    "expert_new": FEATURE_GROUP_EXPERT_NEW_TRIMMED,
}
ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW_TRIMMED = (
    ALL_FEATURES_V5_PRUNED_COLLINEAR + FEATURE_GROUP_EXPERT_NEW_TRIMMED
)

# Second collinear-cluster pass on the adopted 27-feature set itself (not an
# expert-review item -- found by directly checking pairwise Pearson r within
# ALL_FEATURES_V5_PRUNED_COLLINEAR, since RubricNet's per-descriptor transform
# is a plain linear scalar map, hidden_size/num_layers are inert -- so r>0.9
# between two descriptors IS classic linear-regression multicollinearity, the
# same failure mode that justified dropping avg_fret/p90_fret/high_position_ratio
# for fret_entropy earlier). Two pairs exceed that threshold:
#   avg_stretch_velocity_beats <-> p90_stretch_velocity_beats   r=+0.928
#   max_position_shift <-> p95_position_shift_window            r=+0.927
# For each pair, dropped the member with the WEAKER univariate |Spearman rho|
# vs Difficulty (not the "keep windowed/worst-case" heuristic used elsewhere
# in this file -- checked empirically here and the plain aggregate rho was
# tied-or-higher in both pairs: avg_stretch_velocity +0.397 vs p90 +0.391;
# max_position_shift +0.406 vs p95_position_shift_window +0.365).
_COLLINEAR_CLUSTER_V5_PASS2 = {"p90_stretch_velocity_beats", "p95_position_shift_window"}
FEATURE_GROUPS_V5_PRUNED_COLLINEAR2 = {
    group: [f for f in feats if f not in _COLLINEAR_CLUSTER_V5_PASS2]
    for group, feats in FEATURE_GROUPS_V5_PRUNED_COLLINEAR.items()
}
ALL_FEATURES_V5_PRUNED_COLLINEAR2 = (
    FEATURE_GROUPS_V5_PRUNED_COLLINEAR2["lh"] + FEATURE_GROUPS_V5_PRUNED_COLLINEAR2["rh"]
    + FEATURE_GROUPS_V5_PRUNED_COLLINEAR2["global"]
)

# Generic (instrument-agnostic) descriptors imported from the music21 reimplementation of
# the jSymbolic feature set -- the first descriptors in this thesis not designed by hand.
# Motivation and full A/B/C comparison: guitar/JSYMBOLIC_STUDY.md. Combining all 272
# generic features with the 25 guitar descriptors beat the guitar set alone on every
# metric (tau and Spearman gains above one fold-std), but 272 extra inputs are unusable by
# an additive model that must score and sum each one. These four are the generic features
# that carried that gain, and they clear the same r > 0.9 collinearity bar applied in the
# V5 pruning above (max |r| with a kept descriptor: 0.804, js_Duration vs total_notes).
#
#   js_PitchVariety                rho +0.684 -- distinct pitches used; the pitch-space
#                                  analogue of fret_entropy, which measures fretboard space
#   js_Duration                    rho +0.643 -- sounding duration, distinct from note count
#   js_Range                       rho +0.524 -- pitch span between highest and lowest note
#   js_MostCommonPitchPrevalence   rho -0.514 -- share of the single most repeated pitch
#
# Requires features/guitar_descriptors_v5_jsymbolic.csv (from
# guitar/backfill_jsymbolic_descriptors_v5.py), not the plain V5 csv. 11 of the 640 pieces
# have NaN here (music21 parse failures) and are handled by the existing train-fold median
# imputation. As with every other descriptor-set decision after V3, this is to be decided
# on VALIDATION metrics, not test.
JSYMBOLIC_COLUMNS = [
    "js_PitchVariety",
    "js_Duration",
    "js_Range",
    "js_MostCommonPitchPrevalence",
]
FEATURE_GROUPS_V5_PRUNED_COLLINEAR2_JS = {
    **FEATURE_GROUPS_V5_PRUNED_COLLINEAR2,
    "generic": JSYMBOLIC_COLUMNS,
}
ALL_FEATURES_V5_PRUNED_COLLINEAR2_JS = ALL_FEATURES_V5_PRUNED_COLLINEAR2 + JSYMBOLIC_COLUMNS

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
