"""Within source=='pdf' only: does accuracy/error differ between pieces that
have real extracted rhythm (has_rhythm=True) and the dummy uniform-quarter
placeholders (has_rhythm=False)? Reuses guitar/per_source_predictions.csv
(no re-inference needed) joined against the has_rhythm flag in
features/guitar_descriptors_v3.csv.

Writes guitar/figures/accuracy_by_rhythm_v3.png and
guitar/figures/error_by_rhythm_v3.png, one panel per model.
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from guitar.prepare_splits import make_piece_id

MODELS = ["RubricNet", "Fuzzy Complete Search", "Fuzzy Pattern Tree"]
GROUPS = [True, False]
GROUP_LABELS = {True: "has rhythm", False: "no rhythm (dummy)"}
COLORS = {True: "#2a78d6", False: "#eda100"}

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


def load():
    preds = pd.read_csv("guitar/per_source_predictions.csv")
    preds = preds[preds["source"] == "pdf"].copy()

    feats = pd.read_csv("features/guitar_descriptors_v3.csv")
    feats["piece_id"] = feats.apply(make_piece_id, axis=1)
    rhythm_map = feats.set_index("piece_id")["has_rhythm"].to_dict()

    preds["has_rhythm"] = preds["piece_id"].map(rhythm_map)
    preds["abs_err"] = (preds["y_true"] - preds["y_pred"]).abs()
    preds["correct"] = preds["y_true"] == preds["y_pred"]
    return preds


def plot_accuracy(df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        accs = sub.groupby("has_rhythm")["correct"].mean().reindex(GROUPS)
        ns = sub.groupby("has_rhythm")["correct"].count().reindex(GROUPS)
        overall = sub["correct"].mean()

        bars = ax.bar(
            range(len(GROUPS)), accs.values,
            color=[COLORS[g] for g in GROUPS], width=0.5,
        )
        ax.axhline(overall, color="#52514e", linestyle="--", linewidth=1)
        ax.text(1.5, overall, f"pdf overall {overall:.2f}", va="center", ha="left",
                fontsize=8, color="#52514e")

        for bar, n in zip(bars, ns.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={n}", ha="center", va="bottom", fontsize=8, color="#52514e")

        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS])
        ax.set_xlim(-0.6, 2.0)
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Exact-match accuracy")
    fig.suptitle("PDF-source only: accuracy, real rhythm vs. dummy rhythm", y=1.03)
    fig.tight_layout()
    fig.savefig("guitar/figures/accuracy_by_rhythm_v3.png", bbox_inches="tight")
    plt.close(fig)


def plot_error(df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        maes = sub.groupby("has_rhythm")["abs_err"].mean().reindex(GROUPS)
        ns = sub.groupby("has_rhythm")["abs_err"].count().reindex(GROUPS)

        bars = ax.bar(
            range(len(GROUPS)), maes.values,
            color=[COLORS[g] for g in GROUPS], width=0.4, alpha=0.55, zorder=1,
        )
        for i, g in enumerate(GROUPS):
            vals = sub.loc[sub["has_rhythm"] == g, "abs_err"].values
            jitter = rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=10, color=COLORS[g], alpha=0.4, edgecolor="none", zorder=2)

        for bar, n in zip(bars, ns.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={n}", ha="center", va="bottom", fontsize=8, color="#52514e")

        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(model, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("|prediction − truth| (class steps)")
    fig.suptitle("PDF-source only: error, real rhythm vs. dummy rhythm (bars = mean; dots = pieces)", y=1.03)
    fig.tight_layout()
    fig.savefig("guitar/figures/error_by_rhythm_v3.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = load()
    print("=== Accuracy: pdf pieces, has_rhythm vs not ===")
    print(df.groupby(["model", "has_rhythm"])["correct"].agg(["mean", "count"]))
    print("\n=== MAE: pdf pieces, has_rhythm vs not ===")
    print(df.groupby(["model", "has_rhythm"])["abs_err"].agg(["mean", "count"]))
    plot_accuracy(df)
    plot_error(df)
    print("\nWrote guitar/figures/accuracy_by_rhythm_v3.png and guitar/figures/error_by_rhythm_v3.png")
