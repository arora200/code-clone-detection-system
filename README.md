# Code Clone Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/arora200/code-clone-detection-system)

A comprehensive Python-based system for detecting code clones using machine learning and deep learning techniques. This project provides an end-to-end pipeline from data collection and feature engineering to model training, evaluation, and visualization.

## Key Features

-   **Realistic Dataset Generation:** Automatically collects Python projects from GitHub and generates a challenging dataset with mutated code clones (Type 2/3) for robust model evaluation.
-   **Extensive Feature Engineering:** Extracts 34 features from code pairs, including structural metrics, complexity metrics, and Halstead metrics.
-   **Advanced Model Comparison:** Implements and evaluates a wide range of models:
    -   **22 Traditional ML Classifiers** (e.g., ExtraTrees, RandomForest, GradientBoosting).
    -   **Deep Learning Models** like CodeBERT and Simple Neural Networks.
    -   A **Stacking Ensemble (Meta-Classifier)** to combine predictions.
-   **In-depth Analysis:**
    -   Investigates model overfitting using cross-validation and learning curves.
    -   Conducts experiments to find the most robust and efficient model architecture.
-   **Comprehensive Reporting:** Automatically generates a detailed HTML report with performance tables and embedded visualizations, including:
    -   Performance and time complexity comparisons.
    -   Confusion matrices, ROC curves, and feature importance plots.

## Technology Stack

-   **Programming Language:** Python
-   **Machine Learning:** Scikit-learn, PyTorch, Transformers (Hugging Face)
-   **Data Handling:** Pandas, NumPy
-   **Visualization:** Matplotlib, Seaborn
-   **Data Collection:** GitPython, Requests

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arora200/code-clone-detection-system.git
    cd code-clone-detection-system
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the entire pipeline, from data collection to report generation, execute the main script:

```bash
python code_clone_detection_system_reference_code.py
```

The script will:
1.  Collect Python files from the `./data/repos` directory.
2.  Generate a realistic dataset of code clone pairs.
3.  Train and evaluate over 20 different models.
4.  Generate a full HTML report (`analysis_report.html`) with all results and visualizations.

## Project Roadmap

-   [ ] Implement a Graph Neural Network (GNN) model for clone detection.
-   [ ] Perform extensive hyperparameter tuning on the top-performing models.
-   [ ] Expand the code mutation capabilities to generate more complex clone types.
-   [ ] Set up a CI/CD pipeline with GitHub Actions to automate linting and testing.

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) to get started. Also, please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
