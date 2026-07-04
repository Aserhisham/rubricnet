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
from guitar.prepare_splits import ALL_FEATURES_V2, ALL_FEATURES_V3, NUM_CLASSES
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


def compute_metrics(y_true, y_pred):
    res = kendalltau(y_true, y_pred)
    tau = res.correlation if hasattr(res, 'correlation') else res[0]
    if np.isnan(tau):
        tau = 0.0
        
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "acc_plus_minus_1": accuracy_plus_minus_1(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "kendall_tau": float(tau),
    }


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
    args = parser.parse_args()

    if args.v3:
        csv_path = "features/guitar_descriptors_v3.csv"
        columns = ALL_FEATURES_V3
        alias_experiment = "guitar_rubricnet_final_v3"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v3.json"
        out_path = "guitar/rubricnet_results_v3.json"
        print("Training RubricNet on V3 features...")
    else:
        csv_path = "features/guitar_descriptors_v2.csv"
        columns = ALL_FEATURES_V2
        alias_experiment = "guitar_rubricnet_final_v2"
        best_hyperparams_path = "guitar/best_hyperparams_guitar_all_v2.json"
        out_path = "guitar/rubricnet_results_v2.json"
        print("Training RubricNet on V2 features...")

    features, splits = load_data(
        csv_path=csv_path,
        columns=columns
    )
    hyperparams = load_hyperparams(best_hyperparams_path)

    seeds = [0, 1, 2]
    
    # Structure to hold metrics for all seeds and folds
    run_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "acc_plus_minus_1": [],
        "mae": [],
        "mse": [],
        "kendall_tau": [],
        "coarse_3class": {
            "accuracy": [],
            "balanced_accuracy": []
        }
    }

    # Track if any fold collapses below 0.20 accuracy
    collapsed_folds = []

    for seed in seeds:
        print(f"\nTraining with Seed {seed}:")
        pl.seed_everything(seed, workers=True)
        
        seed_acc = []
        seed_bacc = []
        seed_acc1 = []
        seed_mae = []
        seed_mse = []
        seed_tau = []
        
        seed_c_acc = []
        seed_c_bacc = []

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
            args = Args(alias_experiment=alias, **hyperparams)
            
            clf = RubricnetSklearn(
                input_dim=len(columns),
                num_classes=NUM_CLASSES,
                split=split_idx,
                args=args,
                logging=False
            )
            clf.fit(
                scaler.transform(X_train), y_train,
                scaler.transform(X_val), y_val,
                scaler.transform(X_test), y_test,
            )
            clf.load_model(f"checkpoints/{alias}/split_{split_idx}.ckpt")

            y_pred = clf.predict(scaler.transform(X_test)).cpu().numpy()
            
            # Compute 8-class metrics
            fold_m = compute_metrics(y_test, y_pred)
            
            seed_acc.append(fold_m["accuracy"])
            seed_bacc.append(fold_m["balanced_accuracy"])
            seed_acc1.append(fold_m["acc_plus_minus_1"])
            seed_mae.append(fold_m["mae"])
            seed_mse.append(fold_m["mse"])
            seed_tau.append(fold_m["kendall_tau"])

            if fold_m["accuracy"] < 0.20:
                collapsed_folds.append(f"seed_{seed}_split_{split_idx} (acc={fold_m['accuracy']:.4f})")

            # Coarse 3-class evaluation (Phase 7)
            y_test_coarse = map_8_to_3(y_test)
            y_pred_coarse = map_8_to_3(y_pred)
            c_acc = float(accuracy_score(y_test_coarse, y_pred_coarse))
            c_bacc = float(balanced_accuracy_score(y_test_coarse, y_pred_coarse))
            seed_c_acc.append(c_acc)
            seed_c_bacc.append(c_bacc)

            print(f"  Split {split_idx}: acc={fold_m['accuracy']:.4f} bacc={fold_m['balanced_accuracy']:.4f} MAE={fold_m['mae']:.4f}")

        # Store seed lists
        run_metrics["accuracy"].append(seed_acc)
        run_metrics["balanced_accuracy"].append(seed_bacc)
        run_metrics["acc_plus_minus_1"].append(seed_acc1)
        run_metrics["mae"].append(seed_mae)
        run_metrics["mse"].append(seed_mse)
        run_metrics["kendall_tau"].append(seed_tau)
        
        run_metrics["coarse_3class"]["accuracy"].append(seed_c_acc)
        run_metrics["coarse_3class"]["balanced_accuracy"].append(seed_c_bacc)

    # Compute aggregate stats
    metrics_summary = {}
    for key in ["accuracy", "balanced_accuracy", "acc_plus_minus_1", "mae", "mse", "kendall_tau"]:
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

    with open(out_path, "w") as f:
        json.dump({
            "hyperparams": hyperparams,
            "seeds": seeds,
            "metrics": metrics_summary
        }, f, indent=2)
    print(f"\nWrote final detailed results to {out_path}")

    # Generate results table in RESULTS.md
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
