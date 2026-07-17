"""Renders the two per-source diagnostic figures from
guitar/per_source_predictions.csv (produced by guitar/per_source_diagnostics.py):

1. accuracy_by_source_v3.png -- exact-match accuracy per data source, one
   panel per model (RubricNet, Fuzzy Complete Search, Fuzzy Pattern Tree).
2. error_by_source_v3.png -- distribution of |prediction - truth| (in ordinal
   class steps) per data source, one panel per model.

Both use the same fixed source->color mapping and panel order so the two
figures read as one diagnostic pair.

--v5 reads guitar/per_source_predictions_v5.csv (from per_source_diagnostics.py
--v5) and writes *_v5.png figures instead.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["RubricNet", "Fuzzy Complete Search", "Fuzzy Pattern Tree"]
SOURCES = ["pdf", "dada_gp", "gaps"]
SOURCE_LABELS = {"pdf": "PDF (OMR)", "dada_gp": "DadaGP", "gaps": "GAPS"}
# fixed categorical order, slots 1/2/3 from the validated default palette
COLORS = {"pdf": "#2a78d6", "dada_gp": "#008300", "gaps": "#e87ba4"}

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    df["correct"] = df["y_true"] == df["y_pred"]
    return df


def plot_accuracy(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        accs = sub.groupby("source")["correct"].mean().reindex(SOURCES)
        ns = sub.groupby("source")["correct"].count().reindex(SOURCES)
        overall = sub["correct"].mean()

        bars = ax.bar(
            range(len(SOURCES)), accs.values,
            color=[COLORS[s] for s in SOURCES], width=0.6,
        )
        ax.axhline(overall, color="#52514e", linestyle="--", linewidth=1)
        ax.text(2.55, overall, f"overall {overall:.2f}", va="center", ha="left",
                fontsize=8, color="#52514e")

        for bar, n in zip(bars, ns.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={n}", ha="center", va="bottom", fontsize=8, color="#52514e")

        ax.set_xticks(range(len(SOURCES)))
        ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCES])
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Exact-match accuracy")
    fig.suptitle("Accuracy by data source (8-class difficulty, pooled test folds)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_error(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        maes = sub.groupby("source")["abs_err"].mean().reindex(SOURCES)
        ns = sub.groupby("source")["abs_err"].count().reindex(SOURCES)

        bars = ax.bar(
            range(len(SOURCES)), maes.values,
            color=[COLORS[s] for s in SOURCES], width=0.5, alpha=0.55, zorder=1,
        )
        for i, s in enumerate(SOURCES):
            vals = sub.loc[sub["source"] == s, "abs_err"].values
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=10, color=COLORS[s], alpha=0.5, edgecolor="none", zorder=2)

        for bar, n in zip(bars, ns.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={n}", ha="center", va="bottom", fontsize=8, color="#52514e")

        ax.set_xticks(range(len(SOURCES)))
        ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCES])
        ax.set_title(model, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("|prediction − truth| (class steps)")
    fig.suptitle("Prediction error by data source (bars = mean; dots = individual pieces)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5", action="store_true",
                        help="Plot from per_source_predictions_v5.csv into *_v5.png figures")
    args = parser.parse_args()
    tag = "v5" if args.v5 else "v3"
    csv_path = "guitar/per_source_predictions_v5.csv" if args.v5 else "guitar/per_source_predictions.csv"

    df = load(csv_path)
    acc_path = f"guitar/figures/accuracy_by_source_{tag}.png"
    err_path = f"guitar/figures/error_by_source_{tag}.png"
    plot_accuracy(df, acc_path)
    plot_error(df, err_path)
    print(f"Wrote {acc_path} and {err_path}")
