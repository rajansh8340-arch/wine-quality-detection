import pandas as pd
from typing import Tuple
from pathlib import Path
from src.config import DATA_PATH, TARGET_COLUMN, QUALITY_THRESHOLD, TEST_SIZE, RANDOM_STATE

def load_data(filepath: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate dataset from CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    if df.isnull().sum().sum() > 0:
        # Drop or impute missing values if any appear
        df = df.dropna()
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split features and binarize target variable.
    Quality >= 7 is Good (1), otherwise Bad (0).
    """
    X = df.drop(columns=[TARGET_COLUMN])
    # Target binarization
    y = (df[TARGET_COLUMN] >= QUALITY_THRESHOLD).astype(int)
    return X, y

def get_train_test_data(filepath: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and perform Stratified split to preserve class distribution.
    Class imbalance is critical: only ~13.5% of samples are Good wine.
    """
    from sklearn.model_selection import train_test_split
    
    df = load_data(filepath)
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Critical for imbalanced datasets
    )
    return X_train, X_test, y_train, y_test
