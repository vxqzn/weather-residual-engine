from pydantic import BaseModel

class PredictionResponse(BaseModel):
    target_date: str
    raw_forecast_temp_max: float
    predicted_residual_bias: float
    corrected_temp_max: float
    model_id: str
    is_fallback: bool
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_champion_model_id: str | None
    db_connected: bool

class MetricsResponse(BaseModel):
    active_champion_metadata: dict | None
    predictions_served: int
    cache_hits: int
