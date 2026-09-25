import pytest
from fastapi.testclient import TestClient

from twre.service.app import app
from twre.service.engine import ModelEngine

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    response = client.get("/health")
    data = response.json()
    
    assert response.status_code == 200
    assert data["status"] in ["healthy", "not healthy"]
    assert data["model_loaded"] in [True, False]
    assert data["db_connected"] in [True, False]

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    data = response.json()
    
    assert response.status_code == 200
    assert data["active_champion_metadata"] is not None
    assert data["active_champion_metadata"]["model_id"] == ModelEngine().metadata.model_id
    
def test_predict_cached_latency(client):
    warmup_response = client.get("/predict")
    for _ in range(10):
        res = client.get("/predict")
        data = res.json()
        
        assert res.status_code == 200
        assert data["latency_ms"] < 20.0, f"latency above treshold: {data['latency_ms']} ms"
        assert data["is_fallback"] is False

def test_predict_parameter_overrides(client):
    res = client.get("/predict", params={
        "target_date": "2026-01-01",
        "raw_forecast_temp_max": 22.0,
        "error_lag_1": 1.0,
        "rolling_bias_7": 0.5
    })
    
    data = res.json()
    
    assert res.status_code == 200
    assert abs(data["corrected_temp_max"] - (22.0 + data["predicted_residual_bias"])) < 1e-4

def test_fallback_no_model(tmp_path):
    no_engine = ModelEngine(artifacts_dir=tmp_path)
    bias, corrected, model_id, is_fallback = no_engine.predict(
        target_date="2026-01-01",
        raw_forecast_temp_max=21.0,
        error_lag_1=0.0,
        rolling_bias_7=0.0
    )
    
    assert is_fallback is True
    assert bias == 0.0
    assert corrected == 21.0
    assert model_id == "fallback-baseline"