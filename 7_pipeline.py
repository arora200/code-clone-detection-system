"""
Part 7: Main Execution Pipeline
"""

from utils import *
from 1_data_collection import GitHubDataCollector
from 2_feature_engineering import CodeFeatureExtractor
from 5_sota_comparison import StateOfArtComparison
from 6_visualization import create_visualizations

class CodeCloneDetectionPipeline:
    """Main pipeline for the entire code clone detection system"""
    
    def __init__(self):
        self.data_collector = None
        self.feature_extractor = CodeFeatureExtractor()
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
        
        # Save the collected data
        with open('./data/collected_code.json', 'w') as f:
            json.dump(all_code_files, f, indent=4)
            
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

if __name__ == "__main__":
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
