"""Two summary figures for the evaluation chapter.

The chapter's central comparisons are currently carried by dense tables alone.
This produces:

  1. figures/model_comparison.pdf -- every model under the matched protocol on
     accuracy and Kendall's tau, with fold-to-fold standard deviations, so the
     "all inside one standard deviation" reading is visible rather than asserted.
  2. figures/per_band_accuracy.pdf -- RubricNet against the proportional-odds
     baseline by true difficulty band, which the aggregate figures conceal
     entirely.

Usage
-----
    python -m guitar.plot_model_comparison
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "AIM-thesis/figures"

# (label, accuracy, acc_std, tau, tau_std, group)
# Values as reported in Table "Every baseline under one protocol" and the
# RubricNet row of the final generation.
MODELS = [
    ("Majority class",              0.194, 0.003, 0.000, 0.000, "floor"),
    ("Stratified guessing",         0.144, 0.020, 0.016, 0.050, "floor"),
    (r"\texttt{log\_total\_notes}",  0.247, 0.037, 0.543, 0.015, "floor"),
    ("Decision tree",               0.301, 0.037, 0.546, 0.041, "tree"),
    ("Gradient boosting",           0.318, 0.054, 0.582, 0.025, "tree"),
    ("Random forest",               0.318, 0.028, 0.609, 0.023, "tree"),
    ("Extra trees",                 0.320, 0.039, 0.618, 0.024, "tree"),
    ("Multinomial logistic",        0.311, 0.034, 0.594, 0.028, "linear"),
    ("Ordinal logistic (F--H)",     0.347, 0.034, 0.625, 0.022, "linear"),
    ("Ordinal logistic (prop.\\ odds)", 0.383, 0.030, 0.662, 0.024, "linear"),
    ("RubricNet (final)",           0.334, 0.042, 0.640, 0.030, "ours"),
]

BANDS = ["1\u20133", "4\u20135", "6\u20137", "8", "9\u201310", "11\u201312", "13\u201315", "16\u201320"]
RUBRIC_ACC = [0.371, 0.510, 0.258, 0.229, 0.367, 0.343, 0.394, 0.032]
PO_ACC = [0.629, 0.479, 0.532, 0.200, 0.278, 0.171, 0.273, 0.226]
BAND_N = [105, 96, 124, 80, 79, 70, 55, 31]

COLOURS = {"floor": "0.72", "tree": "0.50", "linear": "0.28", "ours": "#1f4e79"}


def model_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    labels = [m[0] for m in MODELS]
    for ax, (vi, si, name, floor) in zip(
        axes, [(1, 2, "Exact accuracy", 0.125), (3, 4, r"Kendall's $\tau$-b", 0.0)]
    ):
        vals = [m[vi] for m in MODELS]
        errs = [m[si] for m in MODELS]
        cols = [COLOURS[m[5]] for m in MODELS]
        ax.errorbar(vals, y, xerr=errs, fmt="o", ms=5, capsize=3, lw=1.1,
                    ecolor="0.6", linestyle="none",
                    markerfacecolor="none", markeredgewidth=0)
        ax.scatter(vals, y, c=cols, s=42, zorder=3)
        ax.set_xlabel(name)
        ax.grid(axis="x", lw=0.4, color="0.88")
        ax.set_axisbelow(True)
        if floor:
            ax.axvline(floor, ls=":", lw=1.0, color="0.55")
            ax.text(floor, len(MODELS) - 0.4, " random", fontsize=7, color="0.45",
                    va="top")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_ylim(-0.8, len(MODELS) - 0.2)
    fig.suptitle("", y=1.0)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "model_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def per_band():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(BANDS))
    w = 0.38
    ax.bar(x - w / 2, RUBRIC_ACC, w, label="RubricNet (final)", color="#1f4e79")
    ax.bar(x + w / 2, PO_ACC, w, label="Ordinal logistic (prop. odds)", color="0.62")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BANDS, BAND_N)])
    ax.set_xlabel("True difficulty band (GuitarBurst grades)", labelpad=6)
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0, 0.68)
    ax.grid(axis="y", lw=0.4, color="0.88")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "per_band_accuracy.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11, "text.usetex": False,
                         "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    MODELS[2] = ("log_total_notes alone",) + MODELS[2][1:]
    MODELS[9] = ("Ordinal logistic (prop. odds)",) + MODELS[9][1:]
    MODELS[8] = ("Ordinal logistic (Frank-Hall)",) + MODELS[8][1:]
    model_comparison()
    per_band()
