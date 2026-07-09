import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitar.baselines import load_data, get_fold_xy
from guitar.prepare_splits import ALL_FEATURES_V3, make_piece_id
from rubricnet.rubricnet import RubricnetSklearn, _prediction2label, map_20_to_8_tensor

def main():
    csv_path = "features/guitar_descriptors_v4.csv"
    columns = ALL_FEATURES_V3
    
    features, splits = load_data(csv_path=csv_path, columns=columns)
    
    # Load raw targets
    df_raw = pd.read_csv(csv_path)
    df_raw["piece_id"] = df_raw.apply(make_piece_id, axis=1)
    raw_difficulty_map = {row["piece_id"]: int(row["Difficulty"]) - 1 for _, row in df_raw.iterrows()}
    
    # Get fold 0
    X_train, y_train = get_fold_xy(features, splits, 0, "train")
    X_val, y_val = get_fold_xy(features, splits, 0, "val")
    X_test, y_test = get_fold_xy(features, splits, 0, "test")
    
    medians = X_train.median().fillna(0.0)
    X_test = X_test.fillna(medians)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load model
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    # Load hyperparams
    from guitar.train_guitar_rubricnet import load_hyperparams
    hyperparams = load_hyperparams("guitar/best_hyperparams_guitar_all_v3.json")
        
    args_cls = Args(alias_experiment="test", **hyperparams)
    
    clf = RubricnetSklearn(
        input_dim=len(columns),
        num_classes=20,
        split=0,
        args=args_cls,
        logging=False
    )
    
    ckpt_path = "checkpoints/guitar_rubricnet_final_v4_raw_seed_0/split_0.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}!")
        return
        
    clf.load_model(ckpt_path)
    
    # Get predictions
    clf.model.eval()
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        probs = clf.model(X_tensor)
        
    print("Model predictions shape:", probs.shape)
    print("First 5 raw probability predictions:")
    print(probs[:5])
    
    pred_labels = _prediction2label(probs)
    print("\nFirst 5 predicted labels (20-class):", pred_labels[:5].tolist())
    
    y_test_fit = np.array([raw_difficulty_map[i] for i in X_test.index])
    print("First 5 true labels (20-class):", y_test_fit[:5].tolist())
    
    # Mapped to 8 classes
    pred_mapped = map_20_to_8_tensor(pred_labels)
    y_test_mapped = map_20_to_8_tensor(torch.tensor(y_test_fit))
    
    print("\nFirst 5 predicted labels (8-class mapped):", pred_mapped[:5].tolist())
    print("First 5 true labels (8-class mapped):", y_test_mapped[:5].tolist())
    print("Original y_test (8-class targets in splits):", y_test.values[:5].tolist())
    
    from sklearn.metrics import accuracy_score
    print("\nAccuracy on test set (8-class mapped):", accuracy_score(y_test_mapped.numpy(), pred_mapped.numpy()))
    print("Accuracy on test set (original y_test):", accuracy_score(y_test.values, pred_mapped.numpy()))

if __name__ == "__main__":
    main()
