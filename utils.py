"""
Utility functions and common imports for the code clone detection system.
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# Traditional ML Models (as per document)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import NuSVC, SVC, LinearSVC
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, RidgeClassifierCV, SGDClassifier,
    Perceptron, PassiveAggressiveClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.calibration import CalibratedClassifierCV

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
