"""
Robust selection script: evaluates top-4 Pareto-best (or top-4 validation accuracy) trials
from the Optuna study across 3 different seeds (0, 1, 2) over 5 folds each (15 runs total per config)
and selects the best one based on mean validation balanced accuracy.
Saves the best hyperparameters to `guitar/best_hyperparams_guitar_all_v2.json`.
"""
import json
import os
import sys
import shutil
from statistics import mean, stdev

os.environ.setdefault("WANDB_MODE", "disabled")

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import torch
import numpy as np
import random
import lightning.pytorch as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from guitar.baselines import get_fold_xy, load_data
from guitar.prepare_splits import ALL_FEATURES_V2, NUM_CLASSES
from rubricnet.rubricnet import RubricnetSklearn


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def set_seed(seed):
    pl.seed_everything(seed, workers=True)


def evaluate_config(trial_num, params, features_data, splits_data):
    """Evaluates a single hyperparameter configuration across 3 seeds and 5 folds (15 evaluations)."""
    n_features = features_data.shape[1]
    val_baccs = []
    
    print(f"\n--- Evaluating Trial {trial_num} ---")
    print(f"Params: {params}")
    
    for seed in (0, 1, 2):
        print(f"  Seed {seed}:")
        set_seed(seed)
        
        seed_baccs = []
        for split in range(5):
            X_train, y_train = get_fold_xy(features_data, splits_data, split, "train")
            X_val, y_val = get_fold_xy(features_data, splits_data, split, "val")
            X_test, y_test = get_fold_xy(features_data, splits_data, split, "test")
            
            scaler = StandardScaler().fit(X_train)
            
            alias_experiment = f"selection_trial_{trial_num}_seed_{seed}"
            args = Args(
                batch_size=params["batch_size"],
                patience=20,
                alias_experiment=alias_experiment,
                weight_decay=params["weight_decay"],
                hidden_size=1,
                num_layers=1,
                dropout=params["dropout"],
                decay_lr=params["decay_lr"],
                lr=params["lr"],
            )
            
            clf = RubricnetSklearn(
                input_dim=n_features,
                num_classes=NUM_CLASSES,
                split=split,
                args=args,
                logging=False,
            )
            
            # Train the model
            clf.fit(
                scaler.transform(X_train), y_train,
                scaler.transform(X_val), y_val,
                scaler.transform(X_test), y_test,
            )
            
            # Load the best checkpoint from training
            ckpt_path = f"checkpoints/{alias_experiment}/split_{split}.ckpt"
            clf.load_model(ckpt_path)
            
            # Predict and score on validation set
            pred_val = clf.predict(scaler.transform(X_val)).cpu().numpy()
            bacc = balanced_accuracy_score(y_val, pred_val)
            seed_baccs.append(bacc)
            val_baccs.append(bacc)
            print(f"    Split {split}: val bacc = {bacc:.4f}")
            
        print(f"  Seed {seed} Mean Val BAcc: {mean(seed_baccs):.4f}")
        
        # Cleanup checkpoints directory for this seed
        shutil.rmtree(f"checkpoints/selection_trial_{trial_num}_seed_{seed}", ignore_errors=True)
        
    mean_val_bacc = mean(val_baccs)
    std_val_bacc = stdev(val_baccs)
    print(f"Trial {trial_num} robust validation bacc over 15 folds: {mean_val_bacc:.4f} +/- {std_val_bacc:.4f}")
    return mean_val_bacc


def main():
    study_name = "guitar_rubricnet_guitar_all_v2"
    sqlite_url = f"sqlite:///guitar/{study_name}.db"
    
    # Load study
    study = optuna.load_study(study_name=study_name, storage=sqlite_url)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Sort completed trials by validation balanced accuracy (value at index 0)
    completed.sort(key=lambda t: t.values[0], reverse=True)
    
    # Take top 4 unique parameter sets (to avoid evaluating exact duplicates if TPESampler resampled)
    unique_candidates = []
    seen_params = set()
    for t in completed:
        p_frozen = frozenset(t.params.items())
        if p_frozen not in seen_params:
            seen_params.add(p_frozen)
            unique_candidates.append(t)
        if len(unique_candidates) == 4:
            break
            
    print(f"Selected {len(unique_candidates)} top candidate trials for robust evaluation:")
    for rank, t in enumerate(unique_candidates):
        print(f"  Rank {rank+1}: Trial {t.number} with val_acc={t.values[0]:.4f}")
        
    # Load V2 features and splits
    features_data, splits_data = load_data(
        csv_path="features/guitar_descriptors_v2.csv",
        columns=ALL_FEATURES_V2,
    )
    
    best_mean_bacc = -1
    best_candidate = None
    
    for t in unique_candidates:
        mean_bacc = evaluate_config(t.number, t.params, features_data, splits_data)
        if mean_bacc > best_mean_bacc:
            best_mean_bacc = mean_bacc
            best_candidate = t
            
    print(f"\n======================================")
    print(f"Best configuration found: Trial {best_candidate.number}")
    print(f"Seed mean validation balanced accuracy: {best_mean_bacc:.4f}")
    print(f"Params: {best_candidate.params}")
    print(f"======================================\n")
    
    # Save the best hyperparams in the expected format
    out_path = "guitar/best_hyperparams_guitar_all_v2.json"
    with open(out_path, "w") as f:
        json.dump({
            "features": "guitar_all_v2",
            "params": best_candidate.params,
            "trial_number": best_candidate.number,
            "seed_mean_val_bacc": best_mean_bacc
        }, f, indent=2)
        
    print(f"Wrote robust best configuration to {out_path}")


if __name__ == "__main__":
    main()
