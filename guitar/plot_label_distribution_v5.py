"""
Label distribution figure for the 640-piece V5 dataset, replacing the old
716-piece figures/label_distribution.pdf in the thesis.

IMPORTANT: unlike the original figure (which reconstructed bin boundaries by
recomputing quantiles on the 716-piece dataframe being plotted), this reuses
the actual frozen `_BIN_EDGES` from guitar/prepare_splits.py directly. The bin
edges are frozen across every descriptor generation including V5 (never
recomputed after V1-V4) -- recomputing quantiles on the 640-piece V5 data
would silently produce DIFFERENT boundaries than what the model was actually
trained on, which would misrepresent the real class definition.

Output: AIM-thesis/figures/label_distribution_v5.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.prepare_splits import _BIN_EDGES

CSV_PATH = "features/guitar_descriptors_v5.csv"
OUT_PATH = "AIM-thesis/figures/label_distribution_v5.pdf"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"{len(df)} pieces, difficulty range {df['Difficulty'].min()}-{df['Difficulty'].max()}")

    # Boundary between consecutive frozen bins: midpoint between one bin's
    # last level and the next bin's first level, e.g. (1-3)|(4-5) -> 3.5.
    boundaries = [(_BIN_EDGES[i][1] + _BIN_EDGES[i + 1][0]) / 2 for i in range(len(_BIN_EDGES) - 1)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Difficulty"], bins=range(1, 22), align="left", color="#4C72B0", edgecolor="white")
    for b in boundaries:
        ax.axvline(b, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("GuitarBurst difficulty (1-20)")
    ax.set_ylabel("# pieces")
    ax.set_title("V5 (640 pieces) label distribution, frozen 8-class bin boundaries (dashed)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    plt.close()

    counts = df["Difficulty"].apply(lambda lvl: next(i for i, (lo, hi) in enumerate(_BIN_EDGES) if lo <= lvl <= hi))
    print("Class sizes:", counts.value_counts().sort_index().to_dict())
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
