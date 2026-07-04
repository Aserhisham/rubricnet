"""
Phase A (v3): re-extract the v2 descriptor set for ALL 716 pieces from a single
unified source -- verified_pieces/all_xmls/ -- using the rhythm-aware timed parser.

This replaces the mixed v2 provenance (PDF-parse for pdf pieces, tokens for dada,
XML for gaps) with one consistent MusicXML parser. It writes the SAME base v2
descriptor columns (so the validation gate can compare like-for-like), plus a
`has_rhythm` flag. Timing-derived (rhythm/context-aware) features come in Phase B.

Output: features/guitar_descriptors_v3.csv (716 rows).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.guitar_features import get_timed_chords_from_xml, calculate_descriptors_v3

ALL_XMLS = "verified_pieces/all_xmls"

RHYTHM_COLUMNS = [
    "max_avg_chord_stretch_window", "p95_position_shift_window", "max_note_density_window",
    "avg_stretch_velocity_beats", "p90_stretch_velocity_beats",
    "avg_position_shift_speed_beats", "max_position_shift_speed_beats",
    "polyphonic_arpeggio_intensity_beats"
]

# Base descriptor columns re-computed from the unified source (v1 + v2 sets).
BASE_COLUMNS = [
    "barre_ratio", "avg_chord_stretch", "max_chord_stretch", "avg_position_shift",
    "fret_change_rate", "arpeggio_density", "avg_string_jump", "max_string_jump",
    "avg_polyphony", "total_notes",
    "log_total_notes", "avg_fret", "p90_fret", "high_position_ratio",
    "open_string_ratio", "p90_chord_stretch", "chord_ratio", "avg_string_span",
    "unique_shape_rate", "shift_rate", "max_position_shift", "std_position_shift",
    "fret_entropy", "string_entropy", "repetition_ratio",
] + RHYTHM_COLUMNS


def resolve_xml(row):
    """Map a CSV row to its file inside all_xmls (by basename of xml_path / file_path)."""
    for col in ("xml_path", "file_path"):
        p = row.get(col)
        if isinstance(p, str) and p:
            cand = os.path.join(ALL_XMLS, os.path.basename(p))
            if os.path.exists(cand):
                return cand
    return None


def main():
    df = pd.read_csv("features/guitar_descriptors.csv")
    print(f"Re-extracting v3 base features for {len(df)} pieces from {ALL_XMLS}/ ...")

    # Guard against two pieces resolving to the same file.
    seen = {}
    records = []
    n_rhythm = n_fail = 0
    for idx, row in df.iterrows():
        path = resolve_xml(row)
        rec = {"idx": idx, "has_rhythm": False}
        if path is None:
            n_fail += 1
            print(f"  [UNRESOLVED] row {idx}: {row['Title'][:40]}")
        else:
            if path in seen:
                print(f"  [COLLISION] row {idx} and {seen[path]} both map to {os.path.basename(path)}")
            seen[path] = idx
            chords, onsets, tempo, tech, has_rhythm = get_timed_chords_from_xml(path)
            if chords is None:
                n_fail += 1
                print(f"  [PARSE-FAIL] row {idx}: {os.path.basename(path)}")
            else:
                feats = calculate_descriptors_v3(chords, onsets, has_rhythm)
                rec.update({c: feats.get(c, np.nan) if c in RHYTHM_COLUMNS else feats.get(c, 0.0) for c in BASE_COLUMNS})
                rec["tempo_bpm"] = tempo if tempo is not None else np.nan
                rec["special_technique_ratio"] = tech if tech is not None else 0.0
                rec["has_rhythm"] = bool(has_rhythm)
                n_rhythm += int(has_rhythm)
        records.append(rec)
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx + 1}/{len(df)} (rhythm so far: {n_rhythm})")

    # Build the extracted-feature frame wholesale (indexed like df), then overwrite.
    feat = pd.DataFrame(records).set_index("idx").reindex(df.index)
    out = df.copy()
    new_cols = BASE_COLUMNS + ["tempo_bpm", "special_technique_ratio", "has_rhythm"]
    for c in new_cols:
        if c in feat.columns:
            # For unresolved/failed rows (NaN in feat), fall back to the v1 value if present.
            fallback = df[c] if c in df.columns else np.nan
            out[c] = feat[c].where(feat[c].notna(), fallback)

    # Impute missing tempo with the global median (fixed constant; noted in thesis).
    tempo_median = out["tempo_bpm"].dropna().median()
    n_missing_tempo = int(out["tempo_bpm"].isna().sum())
    out["tempo_bpm"] = out["tempo_bpm"].fillna(tempo_median)
    out["has_rhythm"] = out["has_rhythm"].fillna(False).astype(bool)

    # No NaN/inf in numeric feature columns.
    for c in BASE_COLUMNS + ["tempo_bpm"]:
        if c in RHYTHM_COLUMNS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            assert (out[c].isna() | np.isfinite(out[c])).all(), f"non-finite/inf in {c}"
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            assert np.isfinite(out[c]).all(), f"non-finite in {c}"

    out.to_csv("features/guitar_descriptors_v3.csv", index=False)
    print(f"\nWrote features/guitar_descriptors_v3.csv ({len(out)} rows)")
    print(f"  has_rhythm: {int(out['has_rhythm'].sum())}/{len(out)}   "
          f"unresolved/parse-fail: {n_fail}   imputed tempo: {n_missing_tempo} (median={tempo_median})")

    # Spearman table (v3 vs difficulty).
    y = out["Difficulty"].values
    corr = []
    for c in BASE_COLUMNS + ["tempo_bpm"]:
        valid_idx = out[c].notna()
        if valid_idx.sum() > 1:
            r = spearmanr(out.loc[valid_idx, c].values, y[valid_idx])[0]
        else:
            r = np.nan
        corr.append((c, r))
    corr = sorted(corr, key=lambda t: -abs(t[1]) if not np.isnan(t[1]) else -1)
    print("\nSpearman |rho| vs Difficulty (v3 features):")
    for c, r in corr:
        print(f"  {c:36s} {r:+.4f}" if not np.isnan(r) else f"  {c:36s} NaN")


if __name__ == "__main__":
    main()
