## Project Overview

This project is a Python-based code clone detection system. It uses machine learning to identify similar code fragments. The system can collect data from GitHub, extract various features from the code (structural, complexity, etc.), and then use these features to train and evaluate different models. The core of the project is a meta-classifier system that combines the predictions of several base classifiers to improve performance. It also includes deep learning models like CodeBERT and GNNs for comparison.

## Key Technologies

*   **Programming Language:** Python
*   **Machine Learning:** Scikit-learn, PyTorch, Transformers (Hugging Face)
*   **Data Handling:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Data Collection:** GitPython, Requests

## Building and Running

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch transformers gitpython requests lightgbm torch-geometric
```

### 2. Run the System

The project can be run directly from the command line.

**Using Synthetic Data (for testing):**

```bash
python code_clone_detection_system_reference_code.py
```

**Using Real GitHub Data:**

To use real data from GitHub, you need to modify the `main` function in `code_clone_detection_system_reference_code.py`:

1.  Set `use_synthetic=False`.
2.  Provide your GitHub token in the `github_token` argument of the `run_full_pipeline` function.

```python
if __name__ == "__main__":
    main()

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
        use_synthetic=False,  # Set to False for real GitHub data
        github_token="YOUR_GITHUB_TOKEN"     # Add your GitHub token here if using real data
    )
    
    # ... (rest of the function)
```

## Development Conventions

*   The code is structured into classes representing different components of the system (e.g., `GitHubDataCollector`, `CodeFeatureExtractor`, `MetaClassifierSystem`).
*   Type hints are used for function signatures.
*   The code is well-commented to explain the implementation details.
*   A random seed is used to ensure reproducibility of the results.
*   The project follows a clear pipeline from data collection to visualization.
