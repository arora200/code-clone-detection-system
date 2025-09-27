import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def get_feature_names():
    # These are the 17 base features from the CodeFeatureExtractor
    base_feature_names = [
        'num_lines', 'num_functions', 'num_classes', 'num_imports',
        'num_variables', 'num_loops', 'num_conditionals', 'max_nesting_depth',
        'avg_line_length', 'num_comments', 'cyclomatic_complexity',
        'num_operators', 'num_operands', 'vocabulary_size', 'program_length',
        'halstead_volume', 'maintainability_index'
    ]
    # The final features are the difference and ratio of these base features for a pair
    feature_names = [f'{name}_diff' for name in base_feature_names] + [f'{name}_ratio' for name in base_feature_names]
    return feature_names

def main():
    print("Generating advanced visualizations...")
    
    # Load data
    try:
        features = np.load('./results/features.npy')
        labels = np.load('./results/labels.npy')
    except FileNotFoundError:
        print("Error: features.npy or labels.npy not found. Please run the main pipeline first.")
        return

    # Impute NaNs just in case
    features = np.nan_to_num(features)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # --- Train the top model ---
    print("Training ExtraTreesClassifier...")
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    os.makedirs("./results/advanced", exist_ok=True)

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Clone', 'Clone'], yticklabels=['Not Clone', 'Clone'])
    plt.title('Confusion Matrix for ExtraTrees Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('./results/advanced/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved.")

    # --- 2. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for ExtraTrees')
    plt.legend(loc="lower right")
    plt.savefig('./results/advanced/roc_curve.png')
    plt.close()
    print("ROC curve saved.")

    # --- 3. Feature Importance ---
    importances = model.feature_importances_
    feature_names = get_feature_names()

    df_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df_importances = df_importances.sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_importances)
    plt.title('Top 15 Feature Importances for ExtraTrees Classifier')
    plt.tight_layout()
    plt.savefig('./results/advanced/feature_importance.png')
    plt.close()
    print("Feature importance plot saved.")

if __name__ == "__main__":
    main()
