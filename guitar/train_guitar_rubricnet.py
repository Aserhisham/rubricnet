import pandas as pd
import numpy as np
import torch
from rubricnet.rubricnet import RubricnetSklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():
    # 1. Load data
    df = pd.read_csv('features/guitar_descriptors.csv')
    
    # 2. Features and Target
    features = [
        'barre_ratio', 'avg_chord_stretch', 'avg_position_shift', 
        'max_position_shift', 'total_position_shift', 'avg_string_jump'
    ]
    
    X = df[features].fillna(0).values
    
    # 3. Bin Difficulty (1-20 -> 0-8)
    def bin_diff(d):
        if d <= 2: return 0
        if d <= 4: return 1
        if d <= 6: return 2
        if d <= 8: return 3
        if d <= 10: return 4
        if d <= 12: return 5
        if d <= 14: return 6
        if d <= 16: return 7
        return 8
        
    y = df['Difficulty'].apply(bin_diff)
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    # 5. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Initialize RubricNet
    class Args:
        def __init__(self):
            self.lr = 0.005
            self.batch_size = 16
            self.hidden_size = 32
            self.num_layers = 1
            self.dropout = 0.05
            self.decay_lr = 0.5
            self.weight_decay = 1e-4
            self.patience = 20
            self.alias_experiment = "guitar_rubricnet"

    args = Args()
    
    # We have 6 features, 9 classes
    model = RubricnetSklearn(input_dim=len(features), num_classes=9, split=0, args=args, logging=False)
    
    # 7. Train
    print("Training Guitar RubricNet...")
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    
    # 8. Evaluate
    y_pred = model.predict(X_test_scaled)
    from sklearn.metrics import accuracy_score, confusion_matrix
    acc = accuracy_score(y_test, y_pred)
    # Convert back to tensor for evaluation helper if needed, but accuracy is fine
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
