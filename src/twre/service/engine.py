import numpy as np
from pathlib import Path
from typing import Any
import time

from twre.models.registry import DEFAULT_ARTIFACTS_DIR, ModelMetadata, load_champion

class ModelEngine:
    def __init__(self, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.champion_path = artifacts_dir / "champion.joblib"
        
        self.model: Any | None = None
        self.metadata: ModelMetadata | None = None
        self._last_mtime: float | None = None
        self.predictions_served: int = 0
        self.cache_hits: int = 0
        self._cached_features: dict[str, float] = {}
        
        self.weights: np.ndarray | None = None
        self.intercept: float | None = None
        
        self.last_reload_check: float = 0.0 
        self._reload_interval_seconds: float = 5.0
        
        self.reload_if_needed(force=True)
    
    def reload_if_needed(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and (now - self.last_reload_check < self._reload_interval_seconds):
            return False
        
        self.last_reload_check = now

        if not self.champion_path.exists():
            self.model = None
            self.metadata = None
            self._last_mtime = None
            self.weights = None
            self.intercept = None
            return False
        
        current_mtime = self.champion_path.stat().st_mtime
        
        if self._last_mtime is None or current_mtime > self._last_mtime:
            self.model, self.metadata = load_champion(self.artifacts_dir)
            if self.model is not None:
                self.weights = np.asarray(self.model.coef_).ravel()
                self.intercept = float(self.model.intercept_)
            else:
                self.weights = None
                self.intercept = None
            
            self._last_mtime = current_mtime
            return True
        
        return False
    
    def predict(
        self, 
        target_date: str | None = None,
        raw_forecast_temp_max: float | None = None,
        error_lag_1: float | None = None,
        rolling_bias_7: float | None = None
    ) -> tuple[float, float, str, bool]:
        self.reload_if_needed()
        self.predictions_served += 1
        
        lag = error_lag_1 or 0.0
        bias = rolling_bias_7 or 0.0
        raw = raw_forecast_temp_max or 0.0
        
        if self.model is None or self.metadata is None or self.weights is None or self.intercept is None:
            return (0.0, raw, "fallback-baseline", True)
        
        residual = (
            self.weights[0] * lag +
            self.weights[1] * bias +
            self.weights[2] * raw +
            self.intercept
        )
        
        corrected_temperature = raw + residual
            
        return (residual, corrected_temperature, self.metadata.model_id, False)
    
    def set_cached_features(self, features: dict[str, float]) -> None:
        self._cached_features = features.copy()
    
    def get_cached_features(self) -> dict[str, float]:
        return self._cached_features.copy()

if __name__ == "__main__":
    engine = ModelEngine()
    engine.predict("2026-09-23", 18.5, 0.2, -0.1)
    
    print(f"Predictions served: {engine.predictions_served}")