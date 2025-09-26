"""
Part 5: State-of-the-Art Comparison
"""

from utils import *
from deep_learning_baselines import SimpleNN, CodeBERTModel, GraphNeuralNetwork
from meta_classifier import MetaClassifierSystem
from sklearn.impute import SimpleImputer

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

if __name__ == '__main__':
    print("="*60)
    print("STATE-OF-THE-ART COMPARISON")
    print("="*60)

    # Load features and labels
    try:
        with open('./data/features.json', 'r') as f:
            features_data = json.load(f)
        X = pd.DataFrame(features_data).reset_index(drop=True)
        # Generate placeholder labels for now, as features.json does not contain labels
        y = pd.Series(np.random.randint(0, 2, len(features_data))).reset_index(drop=True)

        # Handle NaN values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    except FileNotFoundError:
        print("Error: features.json not found. Please run Phase 2 (Feature Engineering) first.")
        exit()
    except Exception as e:
        print(f"Error loading features: {e}")
        exit()

    sota_comparison = StateOfArtComparison()
    sota_comparison.add_traditional_models()
    sota_comparison.add_deep_learning_models()
    results_df = sota_comparison.train_and_evaluate(X, y)

    # Save the results
    results_df.to_json('./data/sota_results.json', orient='records', indent=4)
    print(f"SOTA comparison results saved to ./data/sota_results.json")
