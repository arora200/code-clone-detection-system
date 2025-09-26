# Code Clone Detection System

This project is a comprehensive Python-based system for detecting code clones using machine learning and deep learning techniques. It provides an end-to-end pipeline from data collection and feature engineering to model training, evaluation, and visualization.

## Project Overview

The primary goal of this project is to identify similar or identical code fragments (code clones) within a codebase. It leverages a variety of machine learning models, including a unique meta-classifier approach, to achieve high accuracy in clone detection. The system is designed to be modular and extensible, allowing for easy integration of new models and features.

## Key Features

*   **GitHub Data Collection:** Automatically collects Python CLI projects from GitHub for building a diverse dataset.
*   **Extensive Feature Engineering:** Extracts 17 different metrics from the source code, including:
    *   Structural code metrics (lines of code, number of functions, classes, etc.)
    *   Complexity metrics (cyclomatic complexity, Halstead metrics)
    *   Maintainability Index
    *   Sequence of Code (SoC) features
*   **Deep Learning Baselines:** Implements several deep learning models for comparison, such as:
    *   **CodeBERT:** A transformer-based model pre-trained on a large corpus of code.
    *   **Graph Neural Networks (GNN):** To represent and analyze the code's structure as a graph.
    *   **Simple Neural Network:** A baseline deep learning model.
*   **Meta-Classifier System:** A stacking ensemble model that combines the predictions of multiple base classifiers to improve performance. It follows the methodology described in Algorithm 2 of the associated research document.
*   **State-of-the-Art Comparison:** The system evaluates and compares the performance of 22 different classifiers, including:
    *   Traditional ML models (Random Forest, Gradient Boosting, etc.)
    *   Deep learning models
    *   The meta-classifier ensemble
*   **Comprehensive Visualizations:** Generates a variety of plots and charts to visualize and analyze the results, such as:
    *   Performance metrics comparison charts
    *   Time complexity analysis
    *   Performance heatmaps
    *   Radar charts for multi-metric comparison
    *   Distribution box plots by model type
    *   Learning curves
*   **End-to-End Pipeline:** A complete, runnable pipeline from data collection to results generation.

## Technologies Used

*   **Programming Language:** Python
*   **Machine Learning:** Scikit-learn, PyTorch, Transformers (Hugging Face)
*   **Data Handling:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Data Collection:** GitPython, Requests

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arora200/code-clone-detection-system.git
    cd code-clone-detection-system
    ```

2.  **Install the dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn torch transformers gitpython requests lightgbm torch-geometric
    ```

## Usage

The system can be run in two modes: using synthetic data for a quick test or with real data collected from GitHub.

### Using Synthetic Data

To run the pipeline with synthetic data, execute the following command:

```bash
python code_clone_detection_system_reference_code.py
```

This will run the complete pipeline, train the models, and generate visualizations and a CSV report in the `./results/` directory.

### Using Real GitHub Data

To use real data from GitHub, you need to modify the `main` function in `code_clone_detection_system_reference_code.py`:

1.  Set `use_synthetic=False`.
2.  Provide your GitHub token in the `github_token` argument of the `run_full_pipeline` function. This is recommended to avoid API rate limits.

```python
if __name__ == "__main__":
    pipeline = CodeCloneDetectionPipeline()
    results = pipeline.run_full_pipeline(
        use_synthetic=False,
        github_token="YOUR_GITHUB_TOKEN"  # Add your GitHub token here
    )
```

## Code Structure

The project is organized into a single Python script, `code_clone_detection_system_reference_code.py`, which is divided into the following sections:

*   **Part 1: GitHub Data Collection:** Contains the `GitHubDataCollector` class for cloning repositories and extracting code.
*   **Part 2: Feature Engineering:** The `CodeFeatureExtractor` class is responsible for extracting features from the code.
*   **Part 3: Deep Learning Models:** Implements the `CodeBERTModel` and `GraphNeuralNetwork` models.
*   **Part 4: Meta-Classifier Implementation:** The `MetaClassifierSystem` class builds and evaluates the stacking ensemble.
*   **Part 5: State-of-the-Art Comparison:** The `StateOfArtComparison` class trains and evaluates all the different models.
*   **Part 6: Visualization Functions:** Contains functions for generating plots and charts.
*   **Part 7: Main Execution Pipeline:** The `CodeCloneDetectionPipeline` class orchestrates the entire workflow.
*   **Part 8: Example Usage:** The `main` function, which serves as the entry point of the script.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.
