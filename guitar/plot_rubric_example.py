"""
Per-piece rubric example figure for the adopted V5-pruned-collinear2
(25-feature) RubricNet model, replacing the old rubric_example.pdf (which used
Bach's Chaconne, BWV 1004 -- a piece that turned out to have has_rhythm=False
and is one of the 76 pdf pieces V5 drops, so it no longer exists in this
dataset).

Reuses guitar/descriptor_scores_fold0_v5_pruned_collinear2.csv (already
computed by interpret_rubricnet.py --v5-pruned-collinear2 -- per-descriptor
g_i(x_i) contributions for every fold-0 test-set piece, no re-inference
needed here).

Easy exemplar: Sor's Andantino in C, Op. 35 no. 2 (GuitarBurst level 1,
predicted class 0 correctly) -- same piece as the original V3 figure, still
present in V5 fold 0's test set.
Hard exemplar: J.S. Bach's Fuge in A minor, BWV 1000 (level 17, true class 7,
predicted class 6) -- picked as the closest available surrogate to the
original Chaconne example (same composer, unaccompanied virtuosic solo work,
near-miss-by-one-bin prediction, matching the original narrative structure).

SIGN NOTE: guitar/interpret_rubricnet.py already auto-detects and normalizes
S(x)'s sign convention before saving descriptor_scores_fold0_v5_pruned_collinear2.csv
(this checkpoint happened to converge with S(x) decreasing in difficulty; see
that script's comment), so no additional sign handling is needed here -- the
CSV's g_i(x_i) values already increase with difficulty.

Output: AIM-thesis/figures/rubric_example_v5.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.prepare_splits import ALL_FEATURES_V5_PRUNED_COLLINEAR2, make_piece_id

SCORES_PATH = "guitar/descriptor_scores_fold0_v5_pruned_collinear2.csv"
CSV_PATH = "features/guitar_descriptors_v5.csv"
OUT_PATH = "AIM-thesis/figures/rubric_example_v5.pdf"

EASY_ID = "Andantino in C; Op. 35 no. 2||Fernando Sor"
HARD_ID = "Fuge in A Minor, BWV 1000||J.S. Bach"


def main():
    scores = pd.read_csv(SCORES_PATH, index_col=0)
    df = pd.read_csv(CSV_PATH)
    df["piece_id"] = df.apply(make_piece_id, axis=1)
    meta = df.set_index("piece_id")[["Title", "Composer", "Difficulty"]]

    for pid in (EASY_ID, HARD_ID):
        assert pid in scores.index, f"{pid} not in fold-0 test set of {SCORES_PATH}"

    easy_row = scores.loc[EASY_ID, ALL_FEATURES_V5_PRUNED_COLLINEAR2]
    hard_row = scores.loc[HARD_ID, ALL_FEATURES_V5_PRUNED_COLLINEAR2]
    easy_S = float(easy_row.sum())
    hard_S = float(hard_row.sum())

    order = hard_row.sort_values().index  # sorted by the hard piece's contributions
    easy_vals = easy_row[order].values
    hard_vals = hard_row[order].values

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 9))
    height = 0.38
    ax.barh(y - height / 2, easy_vals, height=height, label=f"{meta.loc[EASY_ID, 'Title']} (easy)", color="#4C72B0")
    ax.barh(y + height / 2, hard_vals, height=height, label=f"{meta.loc[HARD_ID, 'Title']} (hard)", color="#C44E52")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Descriptor contribution $g_i(x_i)$")
    ax.set_title(
        f"Per-descriptor contributions, RubricNet V5-pruned-collinear2 (fold 0)\n"
        f"$S(x)$ = {easy_S:+.2f} (class {int(scores.loc[EASY_ID, 'predicted_label'])}) vs. "
        f"{hard_S:+.2f} (class {int(scores.loc[HARD_ID, 'predicted_label'])})"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    plt.close()

    print(f"Easy: {meta.loc[EASY_ID, 'Title']} ({meta.loc[EASY_ID, 'Composer']}), "
          f"Difficulty={meta.loc[EASY_ID, 'Difficulty']}, true={int(scores.loc[EASY_ID, 'true_label'])}, "
          f"pred={int(scores.loc[EASY_ID, 'predicted_label'])}, S(x)={easy_S:.3f}")
    print(f"Hard: {meta.loc[HARD_ID, 'Title']} ({meta.loc[HARD_ID, 'Composer']}), "
          f"Difficulty={meta.loc[HARD_ID, 'Difficulty']}, true={int(scores.loc[HARD_ID, 'true_label'])}, "
          f"pred={int(scores.loc[HARD_ID, 'predicted_label'])}, S(x)={hard_S:.3f}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
