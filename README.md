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

The system can be run in two ways: as a single, monolithic script or as a series of individual phases.

### Monolithic Execution

To run the entire pipeline from a single script, use the `code_clone_detection_system_reference_code.py` file.

**Using Synthetic Data:**

```bash
python code_clone_detection_system_reference_code.py
```

**Using Real GitHub Data:**

Modify the `main` function in `code_clone_detection_system_reference_code.py`:

1.  Set `use_synthetic=False`.
2.  Provide your GitHub token.

### Phased Execution

The project has been refactored into individual phases, each with its own script and batch file for execution.

### Phase 1: Data Collection

The first phase of the system involves collecting Python source code from GitHub. This is handled by the `1_data_collection.py` script. The process is as follows:

1.  **Search for Repositories:** The script uses the GitHub API to search for popular Python CLI projects. It looks for repositories with a high number of stars to ensure the quality and relevance of the collected code.
2.  **Clone Repositories:** The identified repositories are then cloned into the local `data/repos` directory. To improve efficiency and prevent issues with read-only files, the script now checks if a repository has already been downloaded. If a repository exists, the download is skipped.
3.  **Extract Python Files:** The script traverses the cloned repositories, reads the content of all Python (`.py`) files, and stores them in a JSON file (`data/collected_code.json`). This file serves as the input for the next phase of the pipeline.

To run this phase, execute the following command:

```bash
./run_1_data_collection.bat
```

2.  **Feature Engineering:** `run_2_feature_engineering.bat`
3.  **Deep Learning Baselines:** `run_3_deep_learning_baselines.bat` (Note: This script defines the models and is not meant to be run directly)
4.  **Meta-Classifier:** `run_4_meta_classifier.bat` (Note: This script defines the models and is not meant to be run directly)
5.  **SOTA Comparison:** `run_5_sota_comparison.bat` (Note: This script defines the models and is not meant to be run directly)
6.  **Visualization:** `run_6_visualization.bat` (Note: This script defines the models and is not meant to be run directly)
7.  **Full Pipeline:** `run_7_pipeline.bat`

## Code Structure

The project is organized into the following files:

*   `code_clone_detection_system_reference_code.py`: The original, monolithic script.
*   `utils.py`: Contains utility functions and common imports.
*   `1_data_collection.py`: GitHub data collection.
*   `2_feature_engineering.py`: Feature extraction.
*   `3_deep_learning_baselines.py`: Deep learning model definitions.
*   `4_meta_classifier.py`: Meta-classifier system definition.
*   `5_sota_comparison.py`: State-of-the-art comparison logic.
*   `6_visualization.py`: Visualization functions.
*   `7_pipeline.py`: The main pipeline for phased execution.
*   `run_*.bat`: Batch files for running individual phases.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.