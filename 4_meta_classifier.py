"""
Part 4: Meta-Classifier Implementation
"""

from utils import *

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
