import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status

from twre.service.engine import ModelEngine
from twre.service.schemas import PredictionResponse, HealthResponse, MetricsResponse
from twre.db.session import resolve_connection
from twre.features.pipeline import fetch_training_data, build_feature_matrix

engine = ModelEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.reload_if_needed()
    
    try:
        raw_df = fetch_training_data()
        feature_df = build_feature_matrix(raw_df)
        last_row = feature_df.iloc[-1]

        target_date_str = str(last_row.name.date() if hasattr(last_row.name, "date") else last_row.name)
        engine.set_cached_features({
            "target_date": target_date_str,
            "forecasted_temp_max": last_row["forecasted_temp_max"],
            "error_lag_1": last_row["error_lag_1"],
            "rolling_bias_7": last_row["rolling_bias_7"]
        })
    except Exception as e:
        engine.set_cached_features({
            "target_date": "2026-09-25",
            "forecasted_temp_max": 20.0,
            "error_lag_1": 0.0,
            "rolling_bias_7": 0.0
        })
    yield

app = FastAPI(title="Weather Residual Engine", lifespan=lifespan)  

@app.get("/health", response_model=HealthResponse)
def health_check(response: Response):
    try:
        with resolve_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                db_ok = True
    except Exception as e:
        db_ok = False
        print(f"err during health check ({e})")
    
    model_loaded = engine.model is not None and engine.metadata is not None
    active_champion_model_id = engine.metadata.model_id if engine.metadata else None
    
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status = "unhealthy",
            model_loaded = model_loaded,
            active_champion_model_id = active_champion_model_id,
            db_connected = False
        )

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        active_champion_model_id=active_champion_model_id,
        db_connected = db_ok
    )

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    return MetricsResponse(
        active_champion_metadata=engine.metadata.model_dump() if engine.metadata else None,
        predictions_served=engine.predictions_served,
        cache_hits=engine.cache_hits
    )

@app.get("/predict", response_model=PredictionResponse)
def predict(
    target_date: str | None = None,
    raw_forecast_temp_max: float | None = None,
    error_lag_1: float | None = None,
    rolling_bias_7: float | None = None
):
    t0 = time.perf_counter()
    cached = engine.get_cached_features()
    
    used_cache = False
    
    if target_date is None:
        target_date = str(cached.get("target_date", "2026-09-25"))
        used_cache = True
        
    if raw_forecast_temp_max is None:
        raw_forecast_temp_max = float(cached.get("forecasted_temp_max", 20.0))
        used_cache = True
    
    if error_lag_1 is None:
        error_lag_1 = float(cached.get("error_lag_1", 0.0))
        used_cache = True
    
    if rolling_bias_7 is None:
        rolling_bias_7 = float(cached.get("rolling_bias_7", 0.0))
        used_cache = True
    
    if used_cache:
        engine.cache_hits += 1
        
    prediction_bias, corrected_temp_max, model_id, is_fallback = engine.predict(
        target_date=target_date,
        raw_forecast_temp_max=raw_forecast_temp_max,
        error_lag_1=error_lag_1,
        rolling_bias_7=rolling_bias_7
    )
    
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    return PredictionResponse(
        target_date=target_date,
        raw_forecast_temp_max=raw_forecast_temp_max,
        predicted_residual_bias=prediction_bias,
        corrected_temp_max=corrected_temp_max,
        model_id=model_id,
        is_fallback=is_fallback,
        latency_ms=latency_ms
    )