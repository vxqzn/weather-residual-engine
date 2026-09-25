import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

from twre.models.registry import (
    DEFAULT_ARTIFACTS_DIR,
    ModelMetadata,
    load_ledger,
    load_champion
)
from twre.models.train import FEATURE_COLUMNS

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
        
        self.reload_if_needed()
    
    def reload_if_needed(self) -> bool:
        if not self.champion_path.exists():
            self.model = None
            self.metadata = None
            self._last_mtime = None
            return False
        
        current_mtime = self.champion_path.stat().st_mtime
        
        if self._last_mtime is None or current_mtime > self._last_mtime:
            self.model, self.metadata = load_champion(self.artifacts_dir)
            self._last_mtime = current_mtime
            return True
        return False
    
    def predict(
        self, 
        target_date: str,
        raw_forecast_temp_max: float,
        error_lag_1: float,
        rolling_bias_7: float
    ) -> tuple[float, float, str, bool]:
        self.reload_if_needed()
        self.predictions_served += 1
        
        if self.model is None or self.metadata is None:
            return (0.0, raw_forecast_temp_max, "fallback-baseline", True)
        if self.model is not None and self.metadata is not None:
            X = pd.DataFrame(
                [[error_lag_1, rolling_bias_7, raw_forecast_temp_max]],
                columns=list(FEATURE_COLUMNS)
            )
            prediction_bias = float(self.model.predict(X)[0])
            corrected_temperature = raw_forecast_temp_max + prediction_bias
            
            return (prediction_bias, corrected_temperature, self.metadata.model_id, False)
    
    def set_cached_features(self, features: dict[str, float]) -> None:
        self._cached_features = features.copy()
    
    def get_cached_features(self) -> dict[str, float]:
        return self._cached_features.copy()

if __name__ == "__main__":
    engine = ModelEngine()
    engine.predict("2026-09-23", 18.5, 0.2, -0.1)
    
    print(f"Predictions served: {engine.predictions_served}")