"""Layer the four surviving generic (jSymbolic) descriptors onto the V5 dataset.

Background
----------
`guitar/jsymbolic_comparison.py` found that combining the 25 hand-crafted guitar
descriptors with the generic music21/jSymbolic feature set improved every metric over
the guitar descriptors alone, with the Kendall tau and Spearman gains exceeding one
fold standard deviation. That comparison used 272 generic features, which RubricNet
cannot consume: an additive model scores and sums every input, so a tenfold increase in
input count would destroy both accuracy and the readability of the rubric.

This script takes the narrow version of that finding: the four generic features that
actually earned their place in the combined model's importance ranking, checked against
the same collinearity standard the V5 pruning applied (Pearson r > 0.9 with a kept
descriptor is disqualifying, because each RubricNet subnetwork is a plain affine map).

    feature                        rho vs difficulty   max |r| with the kept 25
    js_PitchVariety                      +0.684        0.764 (log_total_notes)
    js_Duration                          +0.643        0.804 (total_notes)
    js_Range                             +0.524        0.659 (fret_entropy)
    js_MostCommonPitchPrevalence         -0.514        0.519 (fret_entropy)

All four clear the r > 0.9 bar, and `js_PitchVariety`'s correlation with difficulty
(+0.684) is competitive with the strongest existing descriptor (`total_notes`, +0.691).
Conceptually they measure things no guitar descriptor does: variety and span in *pitch*
space rather than fretboard space, and absolute sounding duration rather than note count.

Missing values
--------------
music21 failed to parse 11 of the 640 pieces. Those rows are written as NaN rather than
imputed here, so the existing train-fold median imputation
(`guitar/train_guitar_rubricnet.py:616`) handles them inside each fold -- the same
treatment the rhythm-aware descriptors already receive. Imputing globally in this script
would leak test-fold information into the training statistics.

Note the 11 failures are not uniformly distributed over difficulty (mean class 4.40 vs
the corpus mean 2.81), so these four descriptors are missing more often for hard pieces.
That is a genuine limitation of the generic-feature route and must be reported.

Output: features/guitar_descriptors_v5_jsymbolic.csv (640 rows)
"""

import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import spearmanr

from guitar.prepare_splits import JSYMBOLIC_COLUMNS, make_piece_id

IN_CSV = "features/guitar_descriptors_v5.csv"
JS_CSV = "features/guitar_jsymbolic_v5.csv"
OUT_CSV = "features/guitar_descriptors_v5_jsymbolic.csv"


def main():
    df = pd.read_csv(IN_CSV)
    df["piece_id"] = df.apply(make_piece_id, axis=1)

    js = pd.read_csv(JS_CSV, index_col=0)
    missing_cols = [c for c in JSYMBOLIC_COLUMNS if c not in js.columns]
    if missing_cols:
        raise SystemExit(f"{JS_CSV} is missing expected columns: {missing_cols}")

    merged = df.merge(
        js[JSYMBOLIC_COLUMNS], how="left", left_on="piece_id", right_index=True
    )

    n_missing = merged[JSYMBOLIC_COLUMNS[0]].isna().sum()
    print(f"Rows: {len(merged)} ({n_missing} without jSymbolic features, left as NaN)")

    # Report each descriptor against the raw labels, matching the audit style used for
    # every earlier descriptor generation.
    if "Difficulty" in merged.columns:
        label_col = "Difficulty"
    else:
        label_col = next((c for c in merged.columns if c.lower().startswith("diff")), None)

    if label_col:
        print(f"\nSpearman rho vs {label_col} (non-missing rows only):")
        for c in JSYMBOLIC_COLUMNS:
            ok = merged[[c, label_col]].dropna()
            rho = spearmanr(ok[c], ok[label_col]).statistic
            print(f"  {c:32s} {rho:+.3f}  (n={len(ok)})")

    merged = merged.drop(columns=["piece_id"])
    merged.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(merged)} rows x {merged.shape[1]} cols)")


if __name__ == "__main__":
    main()
