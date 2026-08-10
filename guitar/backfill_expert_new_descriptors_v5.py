"""
Backfill the 7 expert-review descriptor candidates (see guitar/EXPERT_MEETING.md
"New feature candidates surfaced this meeting") onto the V5 dataset (640
pieces): onset_rate_bps, max_onset_rate_bps, chord_change_ratio,
n_meter_changes, irregular_meter_ratio, note_duration_entropy,
finest_subdivision_rank.

Re-parses each piece's MusicXML from verified_pieces/all_xmls/ (same
resolution as guitar/extract_features_v3.py) since these descriptors need
onset/duration/meter data the existing V5 CSV doesn't carry. Writes a NEW csv
(features/guitar_descriptors_v5_expert_new.csv) with all V5 columns plus the
7 new ones -- the original guitar_descriptors_v5.csv is untouched so existing
results stay reproducible.

Output: features/guitar_descriptors_v5_expert_new.csv (640 rows).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.guitar_features import EXPERT_NEW_COLUMNS, calculate_expert_new_descriptors, get_timed_chords_from_xml

ALL_XMLS = "verified_pieces/all_xmls"
IN_CSV = "features/guitar_descriptors_v5.csv"
OUT_CSV = "features/guitar_descriptors_v5_expert_new.csv"


def resolve_xml(row):
    for col in ("xml_path", "file_path"):
        p = row.get(col)
        if isinstance(p, str) and p:
            cand = os.path.join(ALL_XMLS, os.path.basename(p))
            if os.path.exists(cand):
                return cand
    return None


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Backfilling expert-review descriptors for {len(df)} pieces from {ALL_XMLS}/ ...")

    records = []
    n_fail = 0
    for idx, row in df.iterrows():
        path = resolve_xml(row)
        rec = {"idx": idx}
        if path is None:
            n_fail += 1
            print(f"  [UNRESOLVED] row {idx}: {row['Title'][:40]}")
        else:
            chords, onsets, tempo, _tech, has_rhythm = get_timed_chords_from_xml(path)
            if chords is None:
                n_fail += 1
                print(f"  [PARSE-FAIL] row {idx}: {os.path.basename(path)}")
            else:
                # Prefer the already-validated/imputed tempo from the V5 CSV
                # (extract_features_v3.py imputes missing tempo with the
                # dataset median) over a possibly-missing re-parsed value.
                tempo_for_rate = tempo if tempo else row.get("tempo_bpm")
                feats = calculate_expert_new_descriptors(path, chords, onsets, tempo_for_rate, has_rhythm)
                rec.update(feats)
        records.append(rec)
        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{len(df)}")

    feat = pd.DataFrame(records).set_index("idx").reindex(df.index)
    out = df.copy()
    for c in EXPERT_NEW_COLUMNS:
        out[c] = feat[c]

    # Unresolved/parse-fail rows: fall back to neutral defaults rather than
    # dropping rows (matches extract_features_v3.py's fallback discipline).
    defaults = {
        "onset_rate_bps": 0.0, "max_onset_rate_bps": 0.0, "chord_change_ratio": 0.0,
        "n_meter_changes": 0, "irregular_meter_ratio": 0.0,
        "note_duration_entropy": 0.0, "finest_subdivision_rank": 2,
    }
    for c in EXPERT_NEW_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(defaults[c])
        assert np.isfinite(out[c]).all(), f"non-finite in {c}"

    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows, {n_fail} unresolved/parse-fail using defaults)")

    y = out["Difficulty"].values
    print("\nSpearman rho vs Difficulty (new descriptors):")
    for c in EXPERT_NEW_COLUMNS:
        r = spearmanr(out[c].values, y)[0]
        print(f"  {c:28s} {r:+.4f}")


if __name__ == "__main__":
    main()
