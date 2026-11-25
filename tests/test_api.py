import pytest
from api.main import (
    app, load_model_on_startup, read_root, health_check,
    get_model_metrics, predict, predict_batch
)
from src.schema import WineFeatures, WineBatchInput

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    load_model_on_startup()

def test_root_endpoint():
    res = read_root()
    assert "service" in res
    assert res["version"] == "1.0.0"

def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["model_loaded"] is True

def test_metrics_endpoint():
    res = get_model_metrics()
    assert res.accuracy > 0.85
    assert res.roc_auc > 0.85

def test_predict_endpoint_valid_sample():
    sample = WineFeatures(
        fixed_acidity=8.5,
        volatile_acidity=0.28,
        citric_acid=0.56,
        residual_sugar=1.8,
        chlorides=0.092,
        free_sulfur_dioxide=35.0,
        total_sulfur_dioxide=103.0,
        density=0.9969,
        pH=3.3,
        sulphates=0.75,
        alcohol=12.8
    )
    res = predict(sample)
    assert res.is_good_quality is True or res.is_good_quality is False
    assert res.confidence >= 50.0
    assert 0.0 <= res.probability_good <= 1.0
    assert res.key_drivers is not None

def test_batch_prediction_endpoint():
    batch = WineBatchInput(samples=[
        WineFeatures(
            fixed_acidity=8.5, volatile_acidity=0.28, citric_acid=0.56,
            residual_sugar=1.8, chlorides=0.092, free_sulfur_dioxide=35.0,
            total_sulfur_dioxide=103.0, density=0.9969, pH=3.3,
            sulphates=0.75, alcohol=12.8
        ),
        WineFeatures(
            fixed_acidity=7.0, volatile_acidity=0.88, citric_acid=0.04,
            residual_sugar=3.6, chlorides=0.115, free_sulfur_dioxide=8.0,
            total_sulfur_dioxide=85.0, density=0.9984, pH=3.52,
            sulphates=0.48, alcohol=9.2
        )
    ])
    res = predict_batch(batch)
    assert res.total_samples == 2
    assert len(res.predictions) == 2
    assert res.good_quality_count + res.bad_quality_count == 2
