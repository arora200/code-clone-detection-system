import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import ast
from git import Repo
import shutil
import requests
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# Copied necessary classes from the main script for a self-contained analysis
class CodeFeatureExtractor:
    """Extract features from code."""
    def collect_code_structure(self, code_content):
        features = {
            'num_lines': 0, 'num_functions': 0, 'num_classes': 0, 'num_imports': 0,
            'num_variables': 0, 'num_loops': 0, 'num_conditionals': 0, 'max_nesting_depth': 0,
            'avg_line_length': 0, 'num_comments': 0, 'cyclomatic_complexity': 1,
            'num_operators': 0, 'num_operands': 0, 'vocabulary_size': 0, 'program_length': 0,
            'halstead_volume': 0, 'maintainability_index': 0
        }
        try:
            lines = code_content.split('\n')
            features['num_lines'] = len(lines)
            features['avg_line_length'] = np.mean([len(line) for line in lines]) if lines else 0
            features['num_comments'] = sum(1 for line in lines if line.strip().startswith('#'))
            tree = ast.parse(code_content)
            
            class CodeVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.functions = 0; self.classes = 0; self.imports = 0; self.variables = 0
                    self.loops = 0; self.conditionals = 0; self.max_depth = 0; self.current_depth = 0
                    self.operators = 0; self.operands = 0; self.tokens = set()
                def visit_FunctionDef(self, node): self.functions += 1; self.current_depth += 1; self.max_depth = max(self.max_depth, self.current_depth); self.generic_visit(node); self.current_depth -= 1
                def visit_ClassDef(self, node): self.classes += 1; self.current_depth += 1; self.max_depth = max(self.max_depth, self.current_depth); self.generic_visit(node); self.current_depth -= 1
                def visit_Import(self, node): self.imports += 1; self.generic_visit(node)
                def visit_ImportFrom(self, node): self.imports += 1; self.generic_visit(node)
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store): self.variables += 1
                    self.tokens.add(node.id); self.operands += 1; self.generic_visit(node)
                def visit_For(self, node): self.loops += 1; self.generic_visit(node)
                def visit_While(self, node): self.loops += 1; self.generic_visit(node)
                def visit_If(self, node): self.conditionals += 1; self.generic_visit(node)
                def visit_BinOp(self, node): self.operators += 1; self.generic_visit(node)
            
            visitor = CodeVisitor()
            visitor.visit(tree)
            features.update({
                'num_functions': visitor.functions, 'num_classes': visitor.classes, 'num_imports': visitor.imports,
                'num_variables': visitor.variables, 'num_loops': visitor.loops, 'num_conditionals': visitor.conditionals,
                'max_nesting_depth': visitor.max_depth, 'num_operators': visitor.operators, 'num_operands': visitor.operands,
                'vocabulary_size': len(visitor.tokens), 'program_length': visitor.operators + visitor.operands
            })
            if features['vocabulary_size'] > 0 and features['program_length'] > 0:
                features['halstead_volume'] = features['program_length'] * np.log2(features['vocabulary_size'] + 1)
            features['cyclomatic_complexity'] = 1 + visitor.conditionals + visitor.loops
            if features['halstead_volume'] > 0 and features['num_lines'] > 0:
                features['maintainability_index'] = max(0, (171 - 5.2 * np.log(features['halstead_volume']) - 0.23 * features['cyclomatic_complexity'] - 16.2 * np.log(features['num_lines'])) * 100 / 171)
        except Exception: pass
        return features

class CodeCloneDetectionPipeline:
    def __init__(self):
        self.feature_extractor = CodeFeatureExtractor()

    def collect_data_from_local(self, path):
        all_code_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            all_code_files.append({'content': content})
                    except Exception: pass
        return all_code_files

    def create_clone_pairs(self, code_files, num_pairs=1000):
        pairs = []; labels = []
        for i in range(num_pairs // 2):
            if i < len(code_files):
                original = code_files[i]['content']
                pairs.append((original, original)); labels.append(1)
        for i in range(num_pairs // 2):
            if i < len(code_files) - 1:
                idx1 = np.random.randint(0, len(code_files))
                idx2 = np.random.randint(0, len(code_files))
                if idx1 != idx2:
                    pairs.append((code_files[idx1]['content'], code_files[idx2]['content'])); labels.append(0)
        return pairs, labels

    def extract_features(self, pairs):
        all_features = []
        for code1, code2 in pairs:
            features1 = self.feature_extractor.collect_code_structure(code1)
            features2 = self.feature_extractor.collect_code_structure(code2)
            combined_features = []
            for key in features1.keys():
                diff = abs(features1[key] - features2[key])
                combined_features.append(diff)
                if features1[key] + features2[key] > 0:
                    ratio = min(features1[key], features2[key]) / max(features1[key], features2[key] + 0.001)
                else: ratio = 1.0
                combined_features.append(ratio)
            all_features.append(combined_features)
        return np.array(all_features)

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="f1")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")
    return plt

def main():
    print("Investigating overfitting of Traditional ML Models...")
    
    # Load data
    pipeline = CodeCloneDetectionPipeline()
    code_files = pipeline.collect_data_from_local("D:\\AlgorithmBuilding\\Chavi\\data\\repos")
    pairs, labels = pipeline.create_clone_pairs(code_files)
    features = pipeline.extract_features(pairs)
    labels = np.array(labels)
    
    # Impute NaNs in features
    imputer = np.nan_to_num
    features = imputer(features)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Models to investigate
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }
    
    # Cross-validation
    print("\n--- 10-Fold Cross-Validation Results ---")
    for name, model in models.items():
        scores = cross_val_score(model, features_scaled, labels, cv=10, scoring='f1')
        print(f"{name}:")
        print(f"  F1 Score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    # Learning curves
    print("\n--- Generating Learning Curves ---")
    os.makedirs("./results/overfitting_analysis", exist_ok=True)
    
    for name, model in models.items():
        plt.figure()
        plot_learning_curve(model, f"Learning Curve for {name}", features_scaled, labels, cv=5, n_jobs=-1)
        plt.savefig(f"./results/overfitting_analysis/learning_curve_{name}.png")
        plt.close()
        print(f"Learning curve for {name} saved.")

if __name__ == "__main__":
    main()
