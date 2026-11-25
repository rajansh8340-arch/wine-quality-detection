import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "winequality-red.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

# Feature Names in exact order expected by the model
FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

TARGET_COLUMN = "quality"
QUALITY_THRESHOLD = 7  # Quality >= 7 is classified as 'Good' (1), else 'Bad' (0)

# Random seed for reproducible training
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
