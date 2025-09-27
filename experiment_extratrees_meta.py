import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import time
import os

def create_stacking_ensemble(base_models, meta_model, X_train, y_train, X_test):
    # Phase 1: Train base classifiers and create meta features
    n_samples = X_train.shape[0]
    n_base = len(base_models)
    meta_features_train = np.zeros((n_samples, n_base))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Training base models...")
    for i, (name, model) in enumerate(base_models.items()):
        for train_idx, val_idx in kfold.split(X_train):
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val = X_train[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_fold_val)[:, 1]
            else:
                pred = model.predict(X_fold_val)
            meta_features_train[val_idx, i] = pred
            
    # Train final base models on full training data
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        
    # Phase 2: Train meta-classifier
    print("Training meta-classifier...")
    imputer = SimpleImputer(strategy='mean')
    meta_features_train = imputer.fit_transform(meta_features_train)
    meta_model.fit(meta_features_train, y_train)
    
    # Generate predictions for test set
    meta_features_test = np.zeros((X_test.shape[0], n_base))
    for i, (name, model) in enumerate(base_models.items()):
        if hasattr(model, 'predict_proba'):
            meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
        else:
            meta_features_test[:, i] = model.predict(X_test)
            
    meta_features_test = imputer.transform(meta_features_test)
    return meta_model.predict(meta_features_test)

def main():
    print("--- Experiment: ExtraTrees as Meta-Classifier ---")
    
    # Load data
    try:
        X = np.load('./results/features.npy')
        y = np.load('./results/labels.npy')
    except FileNotFoundError:
        print("Error: Data files not found. Please run the main pipeline first.")
        return
        
    X = np.nan_to_num(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # --- Model 1: Single ExtraTrees Classifier ---
    print("\nEvaluating Model 1: Single ExtraTrees Classifier...")
    start_time = time.time()
    single_et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    single_et.fit(X_train, y_train)
    y_pred_single = single_et.predict(X_test)
    time_single = (time.time() - start_time) * 1000
    
    acc_single = accuracy_score(y_test, y_pred_single)
    f1_single = f1_score(y_test, y_pred_single)

    # --- Model 2: Stacking Ensemble with ExtraTrees Meta-Classifier ---
    print("\nEvaluating Model 2: Stacking Ensemble (ExtraTrees Meta)...")
    start_time = time.time()
    
    base_models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    meta_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    
    y_pred_stacking = create_stacking_ensemble(base_models, meta_model, X_train, y_train, X_test)
    time_stacking = (time.time() - start_time) * 1000
    
    acc_stacking = accuracy_score(y_test, y_pred_stacking)
    f1_stacking = f1_score(y_test, y_pred_stacking)

    # --- Comparison ---
    print("\n--- Results ---")
    results = {
        "Model": ["Single ExtraTrees", "Stacking_Ensemble_ET_Meta"],
        "Accuracy": [acc_single, acc_stacking],
        "F1 Score": [f1_single, f1_stacking],
        "Training Time (ms)": [time_single, time_stacking]
    }
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
