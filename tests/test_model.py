import pytest
import numpy as np
from src.model import WineQualityModelPipeline
from src.config import MODEL_PATH, FEATURE_NAMES

@pytest.fixture
def trained_pipeline():
    return WineQualityModelPipeline.load(MODEL_PATH)

def test_pipeline_load(trained_pipeline):
    assert trained_pipeline.model is not None
    assert hasattr(trained_pipeline.model, "predict_proba")

def test_pipeline_single_prediction(trained_pipeline):
    sample = {
        "fixed acidity": 8.5,
        "volatile acidity": 0.28,
        "citric acid": 0.56,
        "residual sugar": 1.8,
        "chlorides": 0.092,
        "free sulfur dioxide": 35.0,
        "total sulfur dioxide": 103.0,
        "density": 0.9969,
        "pH": 3.3,
        "sulphates": 0.75,
        "alcohol": 12.8
    }
    pred_class, prob_bad, prob_good = trained_pipeline.predict_one(sample)
    assert pred_class in [0, 1]
    assert 0.0 <= prob_bad <= 1.0
    assert 0.0 <= prob_good <= 1.0
    assert np.isclose(prob_bad + prob_good, 1.0)
