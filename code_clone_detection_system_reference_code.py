"""
Code Clone Detection System with Deep Learning Baselines and Meta-Classifier
This implementation follows the methodology described in the research document.
"""

import os
import ast
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
from itertools import combinations
from collections import defaultdict

# GitHub data collection
import requests
from git import Repo
import tempfile
import shutil

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                           roc_auc_score, f1_score, classification_report,
                           confusion_matrix, roc_curve, auc)

# Traditional ML Models (as per document)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import NuSVC, SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Try importing optional libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Graph Neural Networks
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("PyTorch Geometric not available. Install with: pip install torch-geometric")

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

#########################
# PART 1: GitHub Data Collection
#########################

class GitHubDataCollector:
    """Collect Python CLI projects from GitHub"""
    
    def __init__(self, github_token=None):
        self.github_token = github_token
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({'Authorization': f'token {github_token}'})
    
    def search_cli_projects(self, query="python cli", max_repos=20):
        """Search for Python CLI projects on GitHub"""
        url = "https://api.github.com/search/repositories"
        params = {
            'q': f'{query} language:python stars:>50',
            'sort': 'stars',
            'per_page': max_repos
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()['items']
        else:
            print(f"Error: {response.status_code}")
            return []
    
    def clone_and_extract_code(self, repo_url, temp_dir="/tmp/repos"):
        """Clone repository and extract Python files"""
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(temp_dir, repo_name)
        
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
            
            Repo.clone_from(repo_url, repo_path)
            
            python_files = []
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            python_files.append({
                                'path': file_path,
                                'content': content,
                                'repo': repo_name
                            })
            
            shutil.rmtree(repo_path)
            return python_files
        except Exception as e:
            print(f"Error cloning {repo_url}: {e}")
            return []

#########################
# PART 2: Feature Engineering
#########################

class CodeFeatureExtractor:
    """Extract features from code as described in the document"""
    
    def __init__(self):
        self.features = []
    
    def collect_code_structure(self, code_content):
        """Implementation of CollectCodeStructure algorithm from the document"""
        features = {
            'num_lines': 0,
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_variables': 0,
            'num_loops': 0,
            'num_conditionals': 0,
            'max_nesting_depth': 0,
            'avg_line_length': 0,
            'num_comments': 0,
            'cyclomatic_complexity': 1,
            'num_operators': 0,
            'num_operands': 0,
            'vocabulary_size': 0,
            'program_length': 0,
            'halstead_volume': 0,
            'maintainability_index': 0
        }
        
        try:
            lines = code_content.split('\n')
            features['num_lines'] = len(lines)
            features['avg_line_length'] = np.mean([len(line) for line in lines]) if lines else 0
            features['num_comments'] = sum(1 for line in lines if line.strip().startswith('#'))
            
            tree = ast.parse(code_content)
            
            class CodeVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.functions = 0
                    self.classes = 0
                    self.imports = 0
                    self.variables = 0
                    self.loops = 0
                    self.conditionals = 0
                    self.max_depth = 0
                    self.current_depth = 0
                    self.operators = 0
                    self.operands = 0
                    self.tokens = set()
                
                def visit_FunctionDef(self, node):
                    self.functions += 1
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_ClassDef(self, node):
                    self.classes += 1
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_Import(self, node):
                    self.imports += 1
                    self.generic_visit(node)
                
                def visit_ImportFrom(self, node):
                    self.imports += 1
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        self.variables += 1
                    self.tokens.add(node.id)
                    self.operands += 1
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self.loops += 1
                    self.generic_visit(node)
                
                def visit_While(self, node):
                    self.loops += 1
                    self.generic_visit(node)
                
                def visit_If(self, node):
                    self.conditionals += 1
                    self.generic_visit(node)
                
                def visit_BinOp(self, node):
                    self.operators += 1
                    self.generic_visit(node)
            
            visitor = CodeVisitor()
            visitor.visit(tree)
            
            features['num_functions'] = visitor.functions
            features['num_classes'] = visitor.classes
            features['num_imports'] = visitor.imports
            features['num_variables'] = visitor.variables
            features['num_loops'] = visitor.loops
            features['num_conditionals'] = visitor.conditionals
            features['max_nesting_depth'] = visitor.max_depth
            features['num_operators'] = visitor.operators
            features['num_operands'] = visitor.operands
            features['vocabulary_size'] = len(visitor.tokens)
            features['program_length'] = visitor.operators + visitor.operands
            
            # Halstead metrics
            if features['vocabulary_size'] > 0 and features['program_length'] > 0:
                features['halstead_volume'] = features['program_length'] * np.log2(features['vocabulary_size'] + 1)
            
            # Cyclomatic complexity (simplified)
            features['cyclomatic_complexity'] = 1 + visitor.conditionals + visitor.loops
            
            # Maintainability Index (simplified version)
            if features['halstead_volume'] > 0 and features['num_lines'] > 0:
                features['maintainability_index'] = max(0, (171 - 5.2 * np.log(features['halstead_volume']) 
                                                           - 0.23 * features['cyclomatic_complexity'] 
                                                           - 16.2 * np.log(features['num_lines'])) * 100 / 171)
            
        except Exception as e:
            print(f"Error parsing code: {e}")
        
        return features
    
    def extract_sequence_features(self, code_content):
        """Extract sequence of code (soc_f) features"""
        tokens = []
        try:
            tree = ast.parse(code_content)
            for node in ast.walk(tree):
                tokens.append(type(node).__name__)
        except:
            pass
        
        # Create n-gram features
        ngram_features = defaultdict(int)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            ngram_features[bigram] += 1
        
        return ngram_features

#########################
# PART 3: Deep Learning Models
#########################

class CodeBERTModel(nn.Module):
    """CodeBERT-based model for code clone detection"""
    
    def __init__(self, model_name='microsoft/codebert-base', num_classes=2):
        super(CodeBERTModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class CodeDataset(Dataset):
    """Dataset for code clone detection"""
    
    def __init__(self, codes, labels, tokenizer, max_length=512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = str(self.codes[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for code clone detection"""
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

#########################
# PART 4: Meta-Classifier Implementation
#########################

class MetaClassifierSystem:
    """Implementation of the Meta-Classifier system as described in the document"""
    
    def __init__(self):
        self.base_classifiers = {}
        self.meta_classifier = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def get_top_classifiers(self):
        """Get top 5 classifiers as mentioned in the document"""
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'LabelPropagation': LabelPropagation(max_iter=1000)
        }
        return classifiers
    
    def get_all_classifiers(self):
        """Get all classifiers mentioned in Table 3.5"""
        classifiers = {
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Bagging': BaggingClassifier(random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'LabelPropagation': LabelPropagation(max_iter=1000),
            'NuSVC': NuSVC(probability=True, random_state=42),
            'KNeighbors': KNeighborsClassifier(),
            'SVC': SVC(probability=True, random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'RidgeCV': RidgeClassifierCV(),
            'Ridge': RidgeClassifier(random_state=42),
            'LDA': LinearDiscriminantAnalysis(),
            'LinearSVC': LinearSVC(random_state=42, max_iter=10000),
            'CalibratedCV': CalibratedClassifierCV(base_estimator=LinearSVC(max_iter=10000)),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'NearestCentroid': NearestCentroid(),
            'SGD': SGDClassifier(random_state=42),
            'QDA': QuadraticDiscriminantAnalysis(),
            'LabelSpreading': LabelSpreading(max_iter=1000),
            'Perceptron': Perceptron(random_state=42),
            'PassiveAggressive': PassiveAggressiveClassifier(random_state=42)
        }
        
        if LIGHTGBM_AVAILABLE:
            classifiers['LGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        return classifiers
    
    def create_stacking_ensemble(self, base_models, meta_model, X_train, y_train, X_test=None):
        """Create stacking ensemble as per Algorithm 2 in the document"""
        n_samples = X_train.shape[0]
        n_base = len(base_models)
        
        # Phase 1: Train base classifiers and create meta features
        meta_features_train = np.zeros((n_samples, n_base))
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(base_models.items()):
            print(f"Training base model: {name}")
            for j, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                
                # Get predictions for validation fold
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_fold_val)[:, 1]
                else:
                    pred = model.predict(X_fold_val)
                
                meta_features_train[val_idx, i] = pred
        
        # Train final base models on full training data
        for name, model in base_models.items():
            model.fit(X_train, y_train)
        
        # Phase 2: Train meta-classifier
        print(f"Training meta-classifier: {type(meta_model).__name__}")
        meta_model.fit(meta_features_train, y_train)
        
        # Generate predictions for test set if provided
        if X_test is not None:
            meta_features_test = np.zeros((X_test.shape[0], n_base))
            for i, (name, model) in enumerate(base_models.items()):
                if hasattr(model, 'predict_proba'):
                    meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
                else:
                    meta_features_test[:, i] = model.predict(X_test)
            
            return meta_model, meta_features_test
        
        return meta_model, meta_features_train
    
    def evaluate_all_combinations(self, X, y):
        """Evaluate all combinations as per the methodology"""
        top_classifiers = self.get_top_classifiers()
        results = []
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Test all combinations of 3 base classifiers + 1 meta classifier
        classifier_names = list(top_classifiers.keys())
        
        for base_combo in combinations(classifier_names, 3):
            for meta_name in classifier_names:
                if meta_name not in base_combo:
                    print(f"\nEvaluating: Base={base_combo}, Meta={meta_name}")
                    
                    base_models = {name: top_classifiers[name] for name in base_combo}
                    meta_model = top_classifiers[meta_name]
                    
                    start_time = time.time()
                    
                    # Create stacking ensemble
                    trained_meta, meta_features_test = self.create_stacking_ensemble(
                        base_models, meta_model, X_train, y_train, X_test
                    )
                    
                    # Make predictions
                    y_pred = trained_meta.predict(meta_features_test)
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, y_pred)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    # Calculate ROC AUC if possible
                    try:
                        if hasattr(trained_meta, 'predict_proba'):
                            y_proba = trained_meta.predict_proba(meta_features_test)[:, 1]
                        else:
                            y_proba = y_pred
                        roc_auc = roc_auc_score(y_test, y_proba)
                    except:
                        roc_auc = 0.0
                    
                    time_taken = (time.time() - start_time) * 1000  # ms
                    
                    results.append({
                        'base_classifiers': ', '.join(base_combo),
                        'meta_classifier': meta_name,
                        'accuracy': acc,
                        'balanced_accuracy': bal_acc,
                        'roc_auc': roc_auc,
                        'f1_score': f1,
                        'time_ms': time_taken
                    })
        
        # Sort by F1 score (as per document emphasis)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        return results_df

#########################
# PART 5: State-of-the-Art Comparison
#########################

class StateOfArtComparison:
    """Compare meta-classifier with state-of-the-art methods"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_traditional_models(self):
        """Add traditional ML models"""
        meta_system = MetaClassifierSystem()
        self.models['Traditional_ML'] = meta_system.get_all_classifiers()
    
    def add_deep_learning_models(self):
        """Add deep learning models"""
        dl_models = {}
        
        # Simple neural network baseline
        class SimpleNN(nn.Module):
            def __init__(self, input_dim, hidden_dim=128):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, 64)
                self.fc3 = nn.Linear(64, 2)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        dl_models['SimpleNN'] = SimpleNN
        dl_models['CodeBERT'] = CodeBERTModel
        
        if GNN_AVAILABLE:
            dl_models['GNN'] = GraphNeuralNetwork
        
        self.models['Deep_Learning'] = dl_models
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train and evaluate all models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = []
        
        # Evaluate traditional ML models
        if 'Traditional_ML' in self.models:
            print("\n=== Evaluating Traditional ML Models ===")
            for name, model in self.models['Traditional_ML'].items():
                print(f"Training {name}...")
                start_time = time.time()
                
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    acc = accuracy_score(y_test, y_pred)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test_scaled)[:, 1]
                        else:
                            y_proba = y_pred
                        roc_auc = roc_auc_score(y_test, y_proba)
                    except:
                        roc_auc = 0.0
                    
                    time_taken = (time.time() - start_time) * 1000
                    
                    results.append({
                        'model': name,
                        'type': 'Traditional ML',
                        'accuracy': acc,
                        'balanced_accuracy': bal_acc,
                        'roc_auc': roc_auc,
                        'f1_score': f1,
                        'time_ms': time_taken
                    })
                except Exception as e:
                    print(f"Error training {name}: {e}")
        
        # Evaluate Meta-Classifier
        print("\n=== Evaluating Meta-Classifier ===")
        meta_system = MetaClassifierSystem()
        meta_results = meta_system.evaluate_all_combinations(X, y)
        
        # Add best meta-classifier result
        if not meta_results.empty:
            best_meta = meta_results.iloc[0]
            results.append({
                'model': f"MetaClassifier ({best_meta['base_classifiers']})",
                'type': 'Meta-Classifier',
                'accuracy': best_meta['accuracy'],
                'balanced_accuracy': best_meta['balanced_accuracy'],
                'roc_auc': best_meta['roc_auc'],
                'f1_score': best_meta['f1_score'],
                'time_ms': best_meta['time_ms']
            })
        
        # Deep Learning models (simplified evaluation due to complexity)
        if 'Deep_Learning' in self.models:
            print("\n=== Evaluating Deep Learning Models ===")
            
            # Simple NN evaluation
            if 'SimpleNN' in self.models['Deep_Learning']:
                print("Training Simple Neural Network...")
                start_time = time.time()
                
                model = self.models['Deep_Learning']['SimpleNN'](X_train.shape[1])
                optimizer = optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.LongTensor(y_train)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                
                # Simple training loop
                model.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    _, predicted = torch.max(outputs, 1)
                    y_pred = predicted.numpy()
                
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                time_taken = (time.time() - start_time) * 1000
                
                results.append({
                    'model': 'SimpleNN',
                    'type': 'Deep Learning',
                    'accuracy': acc,
                    'balanced_accuracy': bal_acc,
                    'roc_auc': 0.0,
                    'f1_score': f1,
                    'time_ms': time_taken
                })
        
        return pd.DataFrame(results)

#########################
# PART 6: Visualization Functions
#########################

def create_visualizations(results_df, save_path="./results"):
    """Create comprehensive visualizations for the comparison"""
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc']
    titles = ['Accuracy', 'Balanced Accuracy', 'F1 Score', 'ROC AUC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric value
        sorted_df = results_df.sort_values(metric, ascending=False).head(10)
        
        # Create bar plot
        bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
        
        # Color bars by model type
        colors = {'Traditional ML': 'blue', 'Meta-Classifier': 'red', 'Deep Learning': 'green'}
        for i, model_type in enumerate(sorted_df['type']):
            bars[i].set_color(colors.get(model_type, 'gray'))
        
        ax.set_xlabel('Model')
        ax.set_ylabel(title)
        ax.set_title(f'Top 10 Models by {title}')
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Time Complexity Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by time
    time_df = results_df.sort_values('time_ms').head(15)
    
    bars = ax.bar(range(len(time_df)), time_df['time_ms'])
    
    # Color by type
    for i, model_type in enumerate(time_df['type']):
        bars[i].set_color(colors.get(model_type, 'gray'))
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Training Time Comparison (Top 15 Fastest Models)')
    ax.set_xticks(range(len(time_df)))
    ax.set_xticklabels(time_df['model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/time_complexity_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of Model Performance
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data for heatmap
    heatmap_data = results_df.set_index('model')[metrics].head(15)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, ax=ax)
    ax.set_title('Performance Metrics Heatmap (Top 15 Models)')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Radar Chart for Top 5 Models
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    top_5 = results_df.nlargest(5, 'f1_score')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    for idx, row in top_5.iterrows():
        values = [row[m] for m in metrics]
        values += [values[0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'][:30])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.upper() for t in titles])
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Models - Multi-Metric Comparison', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/radar_chart_top5.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Box Plot for Model Types
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for box plot
        data_by_type = []
        labels = []
        for model_type in results_df['type'].unique():
            data_by_type.append(results_df[results_df['type'] == model_type][metric].values)
            labels.append(model_type)
        
        bp = ax.boxplot(data_by_type, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], ['blue', 'red', 'green'][:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        ax.set_ylabel(title)
        ax.set_title(f'{title} Distribution by Model Type')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/model_type_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Learning Curves Simulation (for demonstration)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Simulate learning curves for top models
    top_models = results_df.nlargest(5, 'f1_score')
    
    epochs = np.arange(1, 51)
    for idx, row in top_models.iterrows():
        # Simulate a learning curve
        final_score = row['f1_score']
        curve = final_score * (1 - np.exp(-epochs / 10)) + np.random.normal(0, 0.01, len(epochs))
        ax.plot(epochs, curve, label=row['model'][:30], linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Simulated Learning Curves - Top 5 Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"All visualizations saved to {save_path}/")

#########################
# PART 7: Main Execution Pipeline
#########################

class CodeCloneDetectionPipeline:
    """Main pipeline for the entire code clone detection system"""
    
    def __init__(self):
        self.data_collector = None
        self.feature_extractor = CodeFeatureExtractor()
        self.meta_system = MetaClassifierSystem()
        self.comparison_system = StateOfArtComparison()
        self.data = None
        self.features = None
        self.labels = None
    
    def collect_data_from_github(self, github_token=None, num_repos=10):
        """Collect data from GitHub repositories"""
        print("=== Collecting Data from GitHub ===")
        self.data_collector = GitHubDataCollector(github_token)
        
        repos = self.data_collector.search_cli_projects(max_repos=num_repos)
        all_code_files = []
        
        for repo in repos[:num_repos]:
            print(f"Processing repository: {repo['name']}")
            clone_url = repo['clone_url']
            code_files = self.data_collector.clone_and_extract_code(clone_url)
            all_code_files.extend(code_files)
        
        print(f"Collected {len(all_code_files)} Python files")
        return all_code_files
    
    def create_clone_pairs(self, code_files, num_pairs=1000):
        """Create code clone pairs for training"""
        print("=== Creating Code Clone Pairs ===")
        
        pairs = []
        labels = []
        
        # Create positive pairs (clones) - simplified approach
        for i in range(num_pairs // 2):
            if i < len(code_files):
                # Create slight modifications as clones
                original = code_files[i]['content']
                # Simple clone: same code
                pairs.append((original, original))
                labels.append(1)
        
        # Create negative pairs (non-clones)
        for i in range(num_pairs // 2):
            if i < len(code_files) - 1:
                idx1 = np.random.randint(0, len(code_files))
                idx2 = np.random.randint(0, len(code_files))
                if idx1 != idx2:
                    pairs.append((code_files[idx1]['content'], code_files[idx2]['content']))
                    labels.append(0)
        
        return pairs, labels
    
    def extract_features(self, pairs):
        """Extract features from code pairs"""
        print("=== Extracting Features ===")
        
        all_features = []
        
        for code1, code2 in pairs:
            # Extract structural features (qpc_f)
            features1 = self.feature_extractor.collect_code_structure(code1)
            features2 = self.feature_extractor.collect_code_structure(code2)
            
            # Combine features (simplified - using difference and similarity metrics)
            combined_features = []
            for key in features1.keys():
                # Absolute difference
                diff = abs(features1[key] - features2[key])
                combined_features.append(diff)
                
                # Ratio (similarity)
                if features1[key] + features2[key] > 0:
                    ratio = min(features1[key], features2[key]) / max(features1[key], features2[key] + 0.001)
                else:
                    ratio = 1.0
                combined_features.append(ratio)
            
            all_features.append(combined_features)
        
        return np.array(all_features)
    
    def generate_synthetic_data(self, n_samples=2000, n_features=34):
        """Generate synthetic data for demonstration"""
        print("=== Generating Synthetic Data ===")
        
        # Generate synthetic features (17 metrics * 2 for diff and ratio)
        np.random.seed(42)
        
        # Generate two classes with different distributions
        n_class_0 = n_samples // 2
        n_class_1 = n_samples - n_class_0
        
        # Class 0 (non-clones) - higher differences
        X_class_0 = np.random.randn(n_class_0, n_features) * 2 + 3
        y_class_0 = np.zeros(n_class_0)
        
        # Class 1 (clones) - lower differences
        X_class_1 = np.random.randn(n_class_1, n_features) * 1.5 + 1
        y_class_1 = np.ones(n_class_1)
        
        # Combine
        X = np.vstack([X_class_0, X_class_1])
        y = np.hstack([y_class_0, y_class_1])
        
        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        return X, y.astype(int)
    
    def run_full_pipeline(self, use_synthetic=True, github_token=None):
        """Run the complete pipeline"""
        
        if use_synthetic:
            # Use synthetic data for demonstration
            self.features, self.labels = self.generate_synthetic_data()
        else:
            # Collect real data from GitHub
            code_files = self.collect_data_from_github(github_token)
            pairs, labels = self.create_clone_pairs(code_files)
            self.features = self.extract_features(pairs)
            self.labels = np.array(labels)
        
        print(f"\nDataset shape: {self.features.shape}")
        print(f"Class distribution: {np.bincount(self.labels)}")
        
        # Add all model types
        print("\n=== Setting up Models ===")
        self.comparison_system.add_traditional_models()
        self.comparison_system.add_deep_learning_models()
        
        # Train and evaluate all models
        print("\n=== Training and Evaluation ===")
        results = self.comparison_system.train_and_evaluate(
            self.features, self.labels
        )
        
        # Display results
        print("\n=== Results Summary ===")
        print(results.sort_values('f1_score', ascending=False).to_string())
        
        # Create visualizations
        print("\n=== Creating Visualizations ===")
        create_visualizations(results)
        
        # Save results
        results.to_csv("./results/comparison_results.csv", index=False)
        print("\nResults saved to ./results/comparison_results.csv")
        
        return results

#########################
# PART 8: Example Usage
#########################

def main():
    """Main execution function"""
    
    print("="*60)
    print("CODE CLONE DETECTION SYSTEM")
    print("Meta-Classifier with Deep Learning Baselines")
    print("="*60)
    
    # Initialize pipeline
    pipeline = CodeCloneDetectionPipeline()
    
    # Run the full pipeline
    # Set use_synthetic=False and provide github_token to use real GitHub data
    results = pipeline.run_full_pipeline(
        use_synthetic=True,  # Set to False for real GitHub data
        github_token=None     # Add your GitHub token here if using real data
    )
    
    # Print top 5 models
    print("\n" + "="*60)
    print("TOP 5 MODELS BY F1 SCORE:")
    print("="*60)
    top_5 = results.nlargest(5, 'f1_score')[['model', 'type', 'f1_score', 'accuracy', 'time_ms']]
    print(top_5.to_string(index=False))
    
    # Additional analysis
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY BY MODEL TYPE:")
    print("="*60)
    summary = results.groupby('type')[['accuracy', 'f1_score', 'time_ms']].agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'time_ms': ['mean', 'std']
    }).round(4)
    print(summary)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()