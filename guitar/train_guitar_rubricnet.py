"""
Phase 5: final Guitar RubricNet training/evaluation on V2 features.
Evaluates the model over 3 different random seeds (0, 1, 2) on the 5-fold splits
using hyperparameters loaded from `guitar/best_hyperparams_guitar_all_v2.json`.

Outputs:
- `guitar/rubricnet_results_v2.json` containing detailed metrics.
- `guitar/RESULTS.md` containing a comparative table of baselines and RubricNet.
"""
import json
import os
import sys
from statistics import mean, stdev
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
import lightning.pytorch as pl

os.environ.setdefault("WANDB_MODE", "disabled")

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import (
    ALL_FEATURES_V2, ALL_FEATURES_V3, ALL_FEATURES_V3_PRUNED,
    ALL_FEATURES_V3_BASE, ALL_FEATURES_V3_BASE_NEW, ALL_FEATURES_V5_PRUNED,
    ALL_FEATURES_V5_PRUNED_COLLINEAR, ALL_FEATURES_V5_PRUNED_FULL,
    ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW,
    ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW_TRIMMED, ALL_FEATURES_V5_PRUNED_COLLINEAR2,
    ALL_FEATURES_V5_PRUNED_COLLINEAR2_JS,
    NUM_CLASSES, make_piece_id,
)
from rubricnet.rubricnet import RubricnetSklearn


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


DEFAULT_HYPERPARAMS = dict(
    lr=0.005,
    batch_size=16,
    hidden_size=1,
    num_layers=1,
    dropout=0.05,
    decay_lr=0.5,
    weight_decay=1e-4,
    patience=20,
    label_smoothing_temp=0.0,
    num_coarse_classes=None,
    coarse_loss_weight=0.3,
)

ALIAS_EXPERIMENT_V2 = "guitar_rubricnet_final_v2"
BEST_HYPERPARAMS_PATH_V2 = "guitar/best_hyperparams_guitar_all_v2.json"


def load_hyperparams(path):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            tuned = data["params"]
        hyperparams = dict(DEFAULT_HYPERPARAMS)
        hyperparams.update(tuned)
        print(f"Loaded tuned hyperparameters from {path}: {tuned}")
        return hyperparams
    print(f"No tuned hyperparameters found at {path}, using defaults.")
    return dict(DEFAULT_HYPERPARAMS)


def accuracy_plus_minus_1(y_true, y_pred):
    return float(mean([1.0 if abs(t - p) <= 1 else 0.0 for t, p in zip(y_true, y_pred)]))


def map_8_to_3(classes):
    mapped = []
    for c in classes:
        if c in (0, 1, 2):
            mapped.append(0)  # Easy
        elif c in (3, 4, 5):
            mapped.append(1)  # Medium
        elif c in (6, 7):
            mapped.append(2)  # Hard
        else:
            raise ValueError(f"Unknown class {c}")
    return np.array(mapped)


def map_9_to_3(classes):
    """Coarse mapping for the 9-class v6 labeling (paper-comparison run)."""
    mapped = []
    for c in classes:
        if c in (0, 1, 2):
            mapped.append(0)  # Easy
        elif c in (3, 4, 5):
            mapped.append(1)  # Medium
        elif c in (6, 7, 8):
            mapped.append(2)  # Hard
        else:
            raise ValueError(f"Unknown class {c}")
    return np.array(mapped)


def relative_absolute_error_per_class(y_true, y_pred):
    """Per-class RAE, size-weighted into a single total.

    For each true class c: RAE_c = sum|pred-true| (in c) / sum|true - global_mean| (in c),
    where global_mean is the mean of y_true over the whole test set (not the per-class
    mean, which would be 0/0 since every sample in class c has true == c). Classes are
    then combined weighted by their size (n_c / N), so larger classes dominate the total
    the same way a plain (non-macro) error would, while still exposing the per-class
    breakdown for diagnosing which grades the model struggles with.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    global_mean = y_true.mean()

    per_class = {}
    total = 0.0
    for c in sorted(set(y_true.tolist())):
        mask = y_true == c
        numerator = np.abs(y_pred[mask] - y_true[mask]).sum()
        denominator = np.abs(y_true[mask] - global_mean).sum()
        # denominator is 0 only if class c's value coincides exactly with the
        # global mean; such a class contributes no baseline error to explain.
        rae_c = float(numerator / denominator) if denominator != 0 else 0.0
        weight = mask.sum() / n
        per_class[c] = rae_c
        total += weight * rae_c

    return total, per_class


def accuracy_per_class(y_true, y_pred):
    """Per-class accuracy: for each true difficulty level, the fraction of its
    pieces the model got exactly right. Exists alongside per-class RAE so a
    level's error can be read both ways -- how often it's right, and when
    wrong, how far off -- since with an unequal level distribution a single
    overall accuracy is dominated by the common levels.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_class = {}
    for c in sorted(set(y_true.tolist())):
        mask = y_true == c
        per_class[c] = float((y_pred[mask] == y_true[mask]).mean())
    return per_class


def correct_count_and_distance_per_class(y_true, y_pred):
    """Per-class raw correct-piece count (not a fraction) plus mean |pred-true|
    distance over all of that level's pieces (correct guesses count as 0
    distance) -- a third, more literal reading of "how many did it get right,
    and how far off is it" alongside the fraction-based accuracy_per_class and
    the baseline-normalized relative_absolute_error_per_class.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_class = {}
    for c in sorted(set(y_true.tolist())):
        mask = y_true == c
        n_total = int(mask.sum())
        n_correct = int((y_pred[mask] == y_true[mask]).sum())
        mean_distance = float(np.abs(y_pred[mask] - y_true[mask]).mean())
        per_class[c] = {"n_correct": n_correct, "n_total": n_total, "mean_distance": mean_distance}
    return per_class


def compute_metrics(y_true, y_pred):
    res = kendalltau(y_true, y_pred)
    tau = res.correlation if hasattr(res, 'correlation') else res[0]
    if np.isnan(tau):
        tau = 0.0

    rae_total, rae_per_class = relative_absolute_error_per_class(y_true, y_pred)
    acc_per_class = accuracy_per_class(y_true, y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "acc_plus_minus_1": accuracy_plus_minus_1(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rae": rae_total,
        "rae_per_class": rae_per_class,
        "acc_per_class": acc_per_class,
        "kendall_tau": float(tau),
    }


def map_20_to_8_numpy(y):
    y_clamped = np.clip(y, 0, 19).astype(int)
    mapping = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7])
    return mapping[y_clamped]


def write_results_markdown():
    """Writes the comparative RESULTS.md table."""
    # Load V1 baseline results if available
    v1_baselines = {}
    if os.path.exists("guitar/baseline_results.json"):
        with open("guitar/baseline_results.json") as f:
            v1_baselines = json.load(f)
            
    # Load V2 baseline results if available
    v2_baselines = {}
    if os.path.exists("guitar/baseline_results_v2.json"):
        with open("guitar/baseline_results_v2.json") as f:
            v2_baselines = json.load(f)

    # Load V3 baseline results if available
    v3_baselines = {}
    if os.path.exists("guitar/baseline_results_v3.json"):
        with open("guitar/baseline_results_v3.json") as f:
            v3_baselines = json.load(f)
            
    # Load RubricNet V1 results if available
    v1_rubricnet = {}
    if os.path.exists("guitar/rubricnet_results.json"):
        with open("guitar/rubricnet_results.json") as f:
            v1_rubricnet = json.load(f)

    # Load RubricNet V2 results if available
    v2_rubricnet = {}
    if os.path.exists("guitar/rubricnet_results_v2.json"):
        with open("guitar/rubricnet_results_v2.json") as f:
            v2_rubricnet = json.load(f)

    # Load RubricNet V3 results if available
    v3_rubricnet = {}
    if os.path.exists("guitar/rubricnet_results_v3.json"):
        with open("guitar/rubricnet_results_v3.json") as f:
            v3_rubricnet = json.load(f)

    # Load RubricNet V4 results if available
    v4_rubricnet = {}
    if os.path.exists("guitar/rubricnet_results_v4.json"):
        with open("guitar/rubricnet_results_v4.json") as f:
            v4_rubricnet = json.load(f)

    # Load RubricNet V4 Raw results if available
    v4_rubricnet_raw = {}
    if os.path.exists("guitar/rubricnet_results_v4_raw.json"):
        with open("guitar/rubricnet_results_v4_raw.json") as f:
            v4_rubricnet_raw = json.load(f)

    # Compile the table rows
    # Columns: Model | Accuracy | Balanced Acc | Acc+/-1 | MAE | MSE | Kendall Tau
    rows = []
    
    def add_row(name, metrics_dict, is_list=False):
        if not metrics_dict:
            return
        
        cols = [name]
        for key in ["accuracy", "balanced_accuracy", "acc_plus_minus_1", "mae", "mse", "kendall_tau"]:
            if key not in metrics_dict and key == "acc_plus_minus_1":
                # For baseline models where acc+/-1 wasn't calculated, we can set as N/A or compute it later
                cols.append("N/A")
                continue
            if key not in metrics_dict and key == "kendall_tau":
                cols.append("N/A")
                continue
                
            vals = metrics_dict[key]
            if isinstance(vals, dict) and "mean" in vals and "std" in vals:
                cols.append(f"{vals['mean']:.4f} ± {vals['std']:.4f}")
            elif is_list:
                # vals is a list/nested list of values, compute mean and std
                flat_vals = np.array(vals).flatten()
                mean_val = mean(flat_vals)
                std_val = stdev(flat_vals) if len(flat_vals) > 1 else 0.0
                cols.append(f"{mean_val:.4f} ± {std_val:.4f}")
            else:
                if isinstance(vals, list):
                    mean_val = mean(vals)
                    std_val = stdev(vals) if len(vals) > 1 else 0.0
                    cols.append(f"{mean_val:.4f} ± {std_val:.4f}")
                else:
                    cols.append(f"{vals:.4f}")
        rows.append(cols)

    # 1. Ordinal regression V1
    if "ordinal_regression" in v1_baselines:
        add_row("Ordinal regression V1", v1_baselines["ordinal_regression"])
    # 2. Decision Tree V1
    if "decision_tree" in v1_baselines:
        add_row("Decision Tree V1", v1_baselines["decision_tree"])
    # 3. Random Forest V1
    if "random_forest" in v1_baselines:
        add_row("Random Forest V1", v1_baselines["random_forest"])
    # 4. RubricNet V1
    if v1_rubricnet and "metrics" in v1_rubricnet:
        add_row("RubricNet V1", v1_rubricnet["metrics"])
        
    # 5. Ordinal regression V2
    if "ordinal_regression" in v2_baselines:
        add_row("Ordinal regression V2", v2_baselines["ordinal_regression"])
    # 6. Decision Tree V2
    if "decision_tree" in v2_baselines:
        add_row("Decision Tree V2", v2_baselines["decision_tree"])
    # 7. Random Forest V2
    if "random_forest" in v2_baselines:
        add_row("Random Forest V2", v2_baselines["random_forest"])
    # 8. RubricNet V2
    if v2_rubricnet and "metrics" in v2_rubricnet:
        add_row("RubricNet V2 (Ours)", v2_rubricnet["metrics"], is_list=True)

    # 9. Ordinal regression V3
    if "ordinal_regression" in v3_baselines:
        add_row("Ordinal regression V3", v3_baselines["ordinal_regression"])
    # 10. Decision Tree V3
    if "decision_tree" in v3_baselines:
        add_row("Decision Tree V3", v3_baselines["decision_tree"])
    # 11. Random Forest V3
    if "random_forest" in v3_baselines:
        add_row("Random Forest V3", v3_baselines["random_forest"])
    # 12. RubricNet V3
    if v3_rubricnet and "metrics" in v3_rubricnet:
        add_row("RubricNet V3 (Ours)", v3_rubricnet["metrics"], is_list=True)

    # 13. RubricNet V4
    if v4_rubricnet and "metrics" in v4_rubricnet:
        add_row("RubricNet V4 (LH Fixes)", v4_rubricnet["metrics"], is_list=True)
    # 14. RubricNet V4 Raw
    if v4_rubricnet_raw and "metrics" in v4_rubricnet_raw:
        add_row("RubricNet V4 (LH Fixes + 1-20 Raw Target)", v4_rubricnet_raw["metrics"], is_list=True)

    # Coarse 3-class mapping evaluation table (Phase 7)
    coarse_rows = []
    def add_coarse_row(name, metrics_dict):
        if not metrics_dict or "coarse_3class" not in metrics_dict:
            return
        c_metrics = metrics_dict["coarse_3class"]
        c_acc_flat = np.array(c_metrics["accuracy"]).flatten()
        c_bacc_flat = np.array(c_metrics["balanced_accuracy"]).flatten()
        coarse_rows.append([
            name,
            f"{mean(c_acc_flat):.4f} ± {stdev(c_acc_flat):.4f}",
            f"{mean(c_bacc_flat):.4f} ± {stdev(c_bacc_flat):.4f}"
        ])

    if v2_rubricnet and "metrics" in v2_rubricnet:
        add_coarse_row("RubricNet V2 (Coarse 3-class)", v2_rubricnet["metrics"])
    if v3_rubricnet and "metrics" in v3_rubricnet:
        add_coarse_row("RubricNet V3 (Coarse 3-class)", v3_rubricnet["metrics"])
    if v4_rubricnet and "metrics" in v4_rubricnet:
        add_coarse_row("RubricNet V4 (Coarse 3-class)", v4_rubricnet["metrics"])
    if v4_rubricnet_raw and "metrics" in v4_rubricnet_raw:
        add_coarse_row("RubricNet V4 Raw (Coarse 3-class)", v4_rubricnet_raw["metrics"])

    md_content = []
    md_content.append("# Evaluation Results\n")
    md_content.append("## 8-Class Difficulty Estimation Comparison\n")
    md_content.append("| Model | Accuracy | Balanced Acc | Acc ± 1 | MAE | MSE | Kendall τ |")
    md_content.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        md_content.append("| " + " | ".join(r) + " |")
        
    md_content.append("\n## Coarse 3-Class Evaluation (Easy / Medium / Hard)\n")
    md_content.append("| Model | Accuracy | Balanced Acc |")
    md_content.append("| :--- | :---: | :---: |")
    for r in coarse_rows:
        md_content.append("| " + " | ".join(r) + " |")

    # Add space for Phase 6 Interpretability Analysis (will be completed in the next phase)
    md_content.append("\n## Interpretability Analysis\n")
    md_content.append("*(To be completed after running interpretability script)*\n")

    with open("guitar/RESULTS.md", "w") as f:
        f.write("\n".join(md_content))
    print("Wrote comparative RESULTS.md")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Use version 2 features")
    parser.add_argument("--v3", action="store_true", help="Use version 3 features")
    parser.add_argument("--v4", action="store_true", help="Use version 4 features")
    parser.add_argument("--v3-pruned", action="store_true", help="Use version 3 features minus near-zero-signal descriptors")
    parser.add_argument("--v3-base", action="store_true", help="Use V3 base features (unified provenance, no rhythm-aware features)")
    parser.add_argument("--v3-base-new", action="store_true", help="Use V3 base + hand-crafted interaction features")
    parser.add_argument("--v5", action="store_true", help="Use V5 dataset (V3 features, 76 pdf/no-rhythm dummy pieces dropped)")
    parser.add_argument("--v6", action="store_true", help="Paper-comparison run: V5 pieces, 9 equal-width classes, 60/20/20 splits (see prepare_splits_v6.py)")
    parser.add_argument("--v5-pruned", action="store_true", help="V5 dataset minus avg_polyphony/fret_change_rate (expert review, see guitar/EXPERT_MEETING.md)")
    parser.add_argument("--v5-pruned-collinear", action="store_true", help="V5-pruned minus high_position_ratio/avg_fret/p90_fret (collinear w/ fret_entropy, expert review) -- decide on VAL accuracy, not test")
    parser.add_argument("--v5-pruned-full", action="store_true", help="V5-pruned-collinear minus max_string_jump/arpeggio_density/avg_string_jump/chord_ratio/tempo_bpm (full expert-ruled-out set) -- decide on VAL accuracy, not test")
    parser.add_argument("--v5-pruned-collinear-expert-new", action="store_true", help="V5-pruned-collinear (27 feat) + 7 new expert-proposed descriptors (onset rate, chord_change_ratio, meter, note-duration entropy) -- decide on VAL accuracy, not test")
    parser.add_argument("--v5-pruned-collinear-expert-new-trimmed", action="store_true", help="V5-pruned-collinear (27 feat) + 5 new descriptors (drops n_meter_changes/irregular_meter_ratio, weakest by both rho and RF importance) -- decide on VAL accuracy, not test")
    parser.add_argument("--v5-pruned-collinear2", action="store_true", help="V5-pruned-collinear (27 feat) minus p90_stretch_velocity_beats/p95_position_shift_window (r>0.9 with avg_stretch_velocity_beats/max_position_shift) -- decide on VAL accuracy, not test")
    parser.add_argument("--v5-pruned-collinear2-js", action="store_true", help="V5-pruned-collinear2 (25 feat) + 4 generic jSymbolic descriptors (js_PitchVariety, js_Duration, js_Range, js_MostCommonPitchPrevalence) -- decide on VAL accuracy, not test")
    parser.add_argument("--raw-levels", action="store_true", help="Train on raw 1-20 difficulty levels")
    parser.add_argument("--label-smoothing-temp", type=float, default=None,
                        help="Ordinal label smoothing temperature (0 = hard step, e.g. 0.3 = mild smoothing). Only applies to --v3/--v4 runs.")
    parser.add_argument("--coarse-loss-weight", type=float, default=None,
                        help="Enable the auxiliary coarse 3-class head with this loss weight (e.g. 0.3). Only applies to --v3/--v4 runs.")
    args = parser.parse_args()

    is_experimental = bool(args.v3_pruned or args.v3_base or args.v3_base_new or args.v5 or args.v6 or args.v5_pruned or args.v5_pruned_collinear or args.v5_pruned_full or args.v5_pruned_collinear_expert_new or args.v5_pruned_collinear_expert_new_trimmed or args.v5_pruned_collinear2 or args.v5_pruned_collinear2_js or args.label_smoothing_temp or args.coarse_loss_weight)
    coarse_aux_enabled = bool(args.coarse_loss_weight)
    # v6 uses the 9-class paper-comparison labeling; everything else keeps the
    # frozen 8-class binning. Note num_classes=9 activates the num_classes>8
    # code paths in rubricnet.py (final-layer init, sum-decode, softened class
    # weights) -- i.e. exactly the configuration the original paper ran on CIPI.
    num_classes = 9 if args.v6 else NUM_CLASSES
    coarse_map = map_9_to_3 if args.v6 else map_8_to_3

    splits_path = "guitar/guitar_splits.json"
    if args.v5_pruned_collinear2_js:
        csv_path = "features/guitar_descriptors_v5_jsymbolic.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR2_JS
        alias_experiment = "guitar_rubricnet_final_v5_pruned_collinear2_js"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5_pruned_collinear2_js.json"
        print("Training RubricNet on V5 pruned-collinear2 (25 feat) + 4 generic jSymbolic descriptors...")
    elif args.v5_pruned_collinear2:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR2
        alias_experiment = "guitar_rubricnet_final_v5_pruned_collinear2"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5_pruned_collinear2.json"
        print("Training RubricNet on V5 pruned-collinear2 (25 feat, second collinear-cluster pass)...")
    elif args.v5_pruned_collinear_expert_new_trimmed:
        csv_path = "features/guitar_descriptors_v5_expert_new.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW_TRIMMED
        alias_experiment = "guitar_rubricnet_final_v5_pruned_collinear_expert_new_trimmed"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5_pruned_collinear_expert_new_trimmed.json"
        print("Training RubricNet on V5 pruned-collinear + 5 new descriptors (meter-complexity pair dropped)...")
    elif args.v5_pruned_collinear_expert_new:
        csv_path = "features/guitar_descriptors_v5_expert_new.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR_EXPERT_NEW
        alias_experiment = "guitar_rubricnet_final_v5_pruned_collinear_expert_new"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5_pruned_collinear_expert_new.json"
        print("Training RubricNet on V5 pruned-collinear + 7 new expert-proposed descriptors...")
    elif args.v5_pruned_full:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_FULL
        alias_experiment = "guitar_rubricnet_final_v5_pruned_full"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5_pruned_full.json"
        print("Training RubricNet on V5 pruned-full (all remaining expert-ruled-out descriptors dropped)...")
    elif args.v5_pruned_collinear:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED_COLLINEAR
        alias_experiment = "guitar_rubricnet_final_v5_pruned_collinear"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5_pruned_collinear.json"
        print("Training RubricNet on V5 pruned-collinear (high_position_ratio/avg_fret/p90_fret dropped)...")
    elif args.v5_pruned:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V5_PRUNED
        alias_experiment = "guitar_rubricnet_final_v5_pruned"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5_pruned.json"
        print("Training RubricNet on V5 pruned (avg_polyphony/fret_change_rate dropped, expert review)...")
    elif args.v6:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v6.json"
        columns = ALL_FEATURES_V3
        alias_experiment = "guitar_rubricnet_final_v6"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v6.json"
        print("Training RubricNet on V6 (paper-comparison: 9 equal-width classes, 60/20/20 splits)...")
    elif args.v5:
        csv_path = "features/guitar_descriptors_v5.csv"
        splits_path = "guitar/guitar_splits_v5.json"
        columns = ALL_FEATURES_V3
        alias_experiment = "guitar_rubricnet_final_v5"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v5.json"
        out_path = "guitar/rubricnet_results_v5.json"
        print("Training RubricNet on V5 dataset (V3 features, no-rhythm pdf pieces dropped)...")
    elif args.v3_base_new:
        csv_path = "features/guitar_descriptors_v3_base.csv"
        columns = ALL_FEATURES_V3_BASE_NEW
        alias_experiment = "guitar_rubricnet_final_v3_base_new"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v3_base_new.json"
        out_path = "guitar/rubricnet_results_v3_base_new.json"
        print("Training RubricNet on V3 base + interaction features...")
    elif args.v3_base:
        csv_path = "features/guitar_descriptors_v3.csv"
        columns = ALL_FEATURES_V3_BASE
        alias_experiment = "guitar_rubricnet_final_v3_base"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v3_base.json"
        out_path = "guitar/rubricnet_results_v3_base.json"
        print("Training RubricNet on V3 base features (no rhythm)...")
    elif args.v3_pruned:
        csv_path = "features/guitar_descriptors_v3.csv"
        columns = ALL_FEATURES_V3_PRUNED
        alias_experiment = "guitar_rubricnet_final_v3_pruned"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v3.json"
        out_path = "guitar/rubricnet_results_v3_pruned.json"
        print("Training RubricNet on V3 pruned features...")
    elif args.v3:
        csv_path = "features/guitar_descriptors_v3.csv"
        columns = ALL_FEATURES_V3
        alias_experiment = "guitar_rubricnet_final_v3"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v3.json"
        out_path = "guitar/rubricnet_results_v3.json"
        print("Training RubricNet on V3 features...")
    elif args.v4:
        csv_path = "features/guitar_descriptors_v4.csv"
        columns = ALL_FEATURES_V3
        alias_experiment = "guitar_rubricnet_final_v4"
        if args.raw_levels:
            best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v4_raw.json"
        else:
            best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v3.json"
        out_path = "guitar/rubricnet_results_v4.json"
        print("Training RubricNet on V4 features...")
    else:
        csv_path = "features/guitar_descriptors_v2.csv"
        columns = ALL_FEATURES_V2
        alias_experiment = "guitar_rubricnet_final_v2"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v2.json"
        out_path = "guitar/rubricnet_results_v2.json"
        print("Training RubricNet on V2 features...")

    if args.raw_levels:
        alias_experiment = f"{alias_experiment}_raw"
        out_path = out_path.replace(".json", "_raw.json")
        print("Training on raw 1-20 difficulty levels...")

    if args.label_smoothing_temp:
        temp_tag = str(args.label_smoothing_temp).replace(".", "p")
        alias_experiment = f"{alias_experiment}_smooth_{temp_tag}"
        out_path = out_path.replace(".json", f"_smooth_{temp_tag}.json")
        print(f"Using ordinal label smoothing temperature={args.label_smoothing_temp}")

    if args.coarse_loss_weight:
        weight_tag = str(args.coarse_loss_weight).replace(".", "p")
        alias_experiment = f"{alias_experiment}_coarseaux_{weight_tag}"
        out_path = out_path.replace(".json", f"_coarseaux_{weight_tag}.json")
        print(f"Using auxiliary coarse 3-class head with loss weight={args.coarse_loss_weight}")

    features, splits = load_data(
        csv_path=csv_path,
        splits_path=splits_path,
        columns=columns
    )
    hyperparams = load_hyperparams(best_hyperparams_path)
    if args.label_smoothing_temp:
        hyperparams["label_smoothing_temp"] = args.label_smoothing_temp
    if args.coarse_loss_weight:
        hyperparams["num_coarse_classes"] = 3
        hyperparams["coarse_loss_weight"] = args.coarse_loss_weight

    if args.raw_levels:
        df_raw = pd.read_csv(csv_path)
        df_raw["piece_id"] = df_raw.apply(make_piece_id, axis=1)
        raw_difficulty_map = {row["piece_id"]: int(row["Difficulty"]) - 1 for _, row in df_raw.iterrows()}

    seeds = [0, 1, 2]
    
    # Structure to hold metrics for all seeds and folds
    run_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "acc_plus_minus_1": [],
        "mae": [],
        "mse": [],
        "rae": [],
        "kendall_tau": [],
        "coarse_3class": {
            "accuracy": [],
            "balanced_accuracy": []
        }
    }
    # Validation-set metrics, computed from the same best-on-val checkpoint as
    # the test metrics above. Exists so feature-set/design decisions can be
    # made on val (never touching test) instead of reusing the test folds
    # across many candidate variants -- see the methodology note in
    # guitar/EXPERT_FEATURE_REVIEW.md. Test is still reported for the record,
    # but should only be treated as a final number once, for the configuration
    # already chosen on val.
    val_metrics = {"accuracy": [], "balanced_accuracy": [], "mae": [], "kendall_tau": []}

    # Track if any fold collapses below 0.20 accuracy
    collapsed_folds = []

    for seed in seeds:
        print(f"\nTraining with Seed {seed}:")
        pl.seed_everything(seed, workers=True)
        
        import shutil
        checkpoint_dir = f"checkpoints/{alias_experiment}_seed_{seed}"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        
        seed_acc = []
        seed_bacc = []
        seed_acc1 = []
        seed_mae = []
        seed_mse = []
        seed_rae = []
        seed_tau = []
        
        seed_c_acc = []
        seed_c_bacc = []

        seed_val_acc = []
        seed_val_bacc = []
        seed_val_mae = []
        seed_val_tau = []

        for split_idx in range(5):
            X_train, y_train = get_fold_xy(features, splits, split_idx, "train")
            X_val, y_val = get_fold_xy(features, splits, split_idx, "val")
            X_test, y_test = get_fold_xy(features, splits, split_idx, "test")

            # Train-fold median imputation
            medians = X_train.median().fillna(0.0)
            X_train = X_train.fillna(medians)
            X_val = X_val.fillna(medians)
            X_test = X_test.fillna(medians)

            scaler = StandardScaler().fit(X_train)
            alias = f"{alias_experiment}_seed_{seed}"
            args_cls = Args(alias_experiment=alias, **hyperparams)
            
            clf = RubricnetSklearn(
                input_dim=len(columns),
                num_classes=20 if args.raw_levels else num_classes,
                split=split_idx,
                args=args_cls,
                logging=False
            )

            if args.raw_levels:
                y_train_fit = pd.Series([raw_difficulty_map[i] for i in X_train.index], index=X_train.index)
                y_val_fit = pd.Series([raw_difficulty_map[i] for i in X_val.index], index=X_val.index)
                y_test_fit = pd.Series([raw_difficulty_map[i] for i in X_test.index], index=X_test.index)
            else:
                y_train_fit = y_train
                y_val_fit = y_val
                y_test_fit = y_test

            if coarse_aux_enabled:
                clf.fit(
                    scaler.transform(X_train), y_train_fit,
                    scaler.transform(X_val), y_val_fit,
                    scaler.transform(X_test), y_test_fit,
                    y_train_coarse=map_8_to_3(y_train), y_val_coarse=map_8_to_3(y_val),
                    y_test_coarse=map_8_to_3(y_test),
                )
            else:
                clf.fit(
                    scaler.transform(X_train), y_train_fit,
                    scaler.transform(X_val), y_val_fit,
                    scaler.transform(X_test), y_test_fit,
                )
            clf.load_model(f"checkpoints/{alias}/split_{split_idx}.ckpt")

            y_pred = clf.predict(scaler.transform(X_test)).cpu().numpy()
            if args.raw_levels:
                y_pred = map_20_to_8_numpy(y_pred)
            # Guard against ordinal-decode underflow on collapsing folds: a
            # prediction below the lowest class is the lowest class. No-op for
            # runs that never underflow (e.g. v3), so prior results are unchanged.
            y_pred = np.clip(y_pred, 0, num_classes - 1)

            # Validation-set predictions from the same best-on-val checkpoint,
            # for design decisions that must not touch the test folds.
            y_val_pred = clf.predict(scaler.transform(X_val)).cpu().numpy()
            if args.raw_levels:
                y_val_pred = map_20_to_8_numpy(y_val_pred)
            y_val_pred = np.clip(y_val_pred, 0, num_classes - 1)
            fold_val_m = compute_metrics(y_val, y_val_pred)
            seed_val_acc.append(fold_val_m["accuracy"])
            seed_val_bacc.append(fold_val_m["balanced_accuracy"])
            seed_val_mae.append(fold_val_m["mae"])
            seed_val_tau.append(fold_val_m["kendall_tau"])

            # Compute 8-class metrics
            fold_m = compute_metrics(y_test, y_pred)
            
            seed_acc.append(fold_m["accuracy"])
            seed_bacc.append(fold_m["balanced_accuracy"])
            seed_acc1.append(fold_m["acc_plus_minus_1"])
            seed_mae.append(fold_m["mae"])
            seed_mse.append(fold_m["mse"])
            seed_rae.append(fold_m["rae"])
            seed_tau.append(fold_m["kendall_tau"])

            if fold_m["accuracy"] < 0.20:
                collapsed_folds.append(f"seed_{seed}_split_{split_idx} (acc={fold_m['accuracy']:.4f})")

            # Coarse 3-class evaluation (Phase 7)
            y_test_coarse = coarse_map(y_test)
            y_pred_coarse = coarse_map(y_pred)
            c_acc = float(accuracy_score(y_test_coarse, y_pred_coarse))
            c_bacc = float(balanced_accuracy_score(y_test_coarse, y_pred_coarse))
            seed_c_acc.append(c_acc)
            seed_c_bacc.append(c_bacc)

            print(f"  Split {split_idx}: acc={fold_m['accuracy']:.4f} bacc={fold_m['balanced_accuracy']:.4f} MAE={fold_m['mae']:.4f} RAE={fold_m['rae']:.4f} | val: acc={fold_val_m['accuracy']:.4f} bacc={fold_val_m['balanced_accuracy']:.4f}")

        # Store seed lists
        run_metrics["accuracy"].append(seed_acc)
        run_metrics["balanced_accuracy"].append(seed_bacc)
        run_metrics["acc_plus_minus_1"].append(seed_acc1)
        run_metrics["mae"].append(seed_mae)
        run_metrics["mse"].append(seed_mse)
        run_metrics["rae"].append(seed_rae)
        run_metrics["kendall_tau"].append(seed_tau)

        run_metrics["coarse_3class"]["accuracy"].append(seed_c_acc)
        run_metrics["coarse_3class"]["balanced_accuracy"].append(seed_c_bacc)

        val_metrics["accuracy"].append(seed_val_acc)
        val_metrics["balanced_accuracy"].append(seed_val_bacc)
        val_metrics["mae"].append(seed_val_mae)
        val_metrics["kendall_tau"].append(seed_val_tau)

    # Compute aggregate stats
    metrics_summary = {}
    for key in ["accuracy", "balanced_accuracy", "acc_plus_minus_1", "mae", "mse", "rae", "kendall_tau"]:
        flat_vals = np.array(run_metrics[key]).flatten()
        metrics_summary[key] = {
            "per_fold_per_seed": run_metrics[key],
            "mean": float(mean(flat_vals)),
            "std": float(stdev(flat_vals))
        }
        
    # Coarse 3-class summary
    metrics_summary["coarse_3class"] = {
        "accuracy": run_metrics["coarse_3class"]["accuracy"],
        "balanced_accuracy": run_metrics["coarse_3class"]["balanced_accuracy"]
    }

    val_metrics_summary = {}
    for key in ["accuracy", "balanced_accuracy", "mae", "kendall_tau"]:
        flat_vals = np.array(val_metrics[key]).flatten()
        val_metrics_summary[key] = {
            "per_fold_per_seed": val_metrics[key],
            "mean": float(mean(flat_vals)),
            "std": float(stdev(flat_vals))
        }

    with open(out_path, "w") as f:
        json.dump({
            "hyperparams": hyperparams,
            "seeds": seeds,
            "metrics": metrics_summary,
            "val_metrics": val_metrics_summary
        }, f, indent=2)
    print(f"\nWrote final detailed results to {out_path}")

    print("\n--- Validation Summary (for feature-set/design decisions -- do not use test for this) ---")
    for key, info in val_metrics_summary.items():
        print(f"  {key:18s} {info['mean']:.4f} +/- {info['std']:.4f}")

    # Generate results table in RESULTS.md (skipped for experimental feature
    # sets/hyperparams: write_results_markdown() overwrites the manually
    # curated Interpretability Analysis section unconditionally, so only the
    # canonical v2/v3 runs should trigger it).
    if is_experimental:
        print(f"Experimental run: not touching guitar/RESULTS.md (see {out_path} for numbers)")
    else:
        write_results_markdown()

    # Print summary output to terminal
    print("\n--- Final Results Summary (Mean +/- Std over 15 runs) ---")
    for key, info in metrics_summary.items():
        if key == "coarse_3class":
            continue
        print(f"  {key:18s} {info['mean']:.4f} +/- {info['std']:.4f}")

    if collapsed_folds:
        print(f"\nWARNING: Some folds collapsed below 0.20 accuracy:")
        for cf in collapsed_folds:
            print(f"  {cf}")
    else:
        print(f"\nAcceptance Check: All folds stayed above 0.20 accuracy successfully.")


if __name__ == "__main__":
    main()
