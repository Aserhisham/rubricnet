"""
Option A, Phase 1a/2a: build the "v3 base + interaction" feature table.

The v3 base descriptor set is simply the v2/unified descriptors WITHOUT the eight
rhythm-aware window/velocity features (that split is expressed in prepare_splits
via ALL_FEATURES_V3_BASE). We do not recompute those base columns -- they are
already present, computed once from the unified XML source, in
`features/guitar_descriptors_v3.csv`.

On top of that base, this script appends the five hand-crafted *interaction*
descriptors from `calculate_interaction_descriptors_v3` (INTERACTION_COLUMNS).
Four of them are pure functions of existing base columns; the fifth
(`stretch_under_time_pressure`) also needs the minimum inter-onset interval, so
we re-parse each piece's MusicXML to recover onsets. Pieces that cannot be
resolved/parsed fall back to a default interval (handled inside the descriptor).

Output: features/guitar_descriptors_v3_base.csv (716 rows = v3 columns + 5 new).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.guitar_features import (
    get_timed_chords_from_xml,
    calculate_interaction_descriptors_v3,
    INTERACTION_COLUMNS,
)

ALL_XMLS = "verified_pieces/all_xmls"
V3_CSV = "features/guitar_descriptors_v3.csv"
OUT_CSV = "features/guitar_descriptors_v3_base.csv"

# Base descriptors the interaction features read from (must exist in the v3 CSV).
REQUIRED_BASE = [
    "barre_ratio", "std_position_shift", "p90_chord_stretch", "string_entropy",
    "open_string_ratio", "max_position_shift", "arpeggio_density", "avg_chord_stretch",
]


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
    df = pd.read_csv(V3_CSV)
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    assert not missing, f"v3 CSV missing base columns needed for interactions: {missing}"
    print(f"Building v3 base + interaction features for {len(df)} pieces...")

    n_resolved = n_parsefail = n_norhythm = 0
    records = []
    for idx, row in df.iterrows():
        onsets, has_rhythm = None, False
        path = resolve_xml(row)
        if path is not None:
            chords, ons, tempo, tech, hr = get_timed_chords_from_xml(path)
            if chords is None:
                n_parsefail += 1
            else:
                onsets, has_rhythm = ons, bool(hr)
                n_resolved += 1
                if not hr:
                    n_norhythm += 1
        feats = calculate_interaction_descriptors_v3(row.to_dict(), onsets, has_rhythm)
        records.append(feats)
        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{len(df)} (resolved: {n_resolved})")

    inter = pd.DataFrame(records, index=df.index)
    out = df.copy()
    for c in INTERACTION_COLUMNS:
        out[c] = pd.to_numeric(inter[c], errors="coerce").fillna(0.0)
        assert np.isfinite(out[c]).all(), f"non-finite in {c}"

    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows, +{len(INTERACTION_COLUMNS)} interaction cols)")
    print(f"  resolved: {n_resolved}   parse-fail: {n_parsefail}   "
          f"no-rhythm (default interval used): {n_norhythm}")

    # Spearman |rho| of the new interaction descriptors vs raw Difficulty.
    y = out["Difficulty"].values
    print("\nSpearman rho vs Difficulty (new interaction descriptors):")
    for c in INTERACTION_COLUMNS:
        r = spearmanr(out[c].values, y)[0]
        flag = "  <-- weak" if abs(r) < 0.15 else ""
        print(f"  {c:32s} {r:+.4f}{flag}")


if __name__ == "__main__":
    main()
