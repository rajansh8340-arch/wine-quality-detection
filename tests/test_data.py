import pytest
import pandas as pd
from src.data import load_data, preprocess_data, get_train_test_data
from src.config import FEATURE_NAMES, TARGET_COLUMN

def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 1599
    assert df.isnull().sum().sum() == 0

def test_preprocess_data():
    df = load_data()
    X, y = preprocess_data(df)
    
    assert list(X.columns) == FEATURE_NAMES
    assert TARGET_COLUMN not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)

def test_stratified_split():
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    # 80-20 split check
    assert len(X_train) == 1279
    assert len(X_test) == 320
    
    # Verify stratification maintains positive class proportion within ~1%
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.02
