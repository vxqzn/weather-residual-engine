import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from twre.features.pipeline import build_feature_matrix, fetch_training_data
from twre.models.train import split_train_holdout, train_candidate


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """
    returns
        - model_mae:
        - baseline_mae:
        - mae_delta: (model_mae - baseline_mae)
    """
    y_pred = model.predict(X)
    model_mae = mean_absolute_error(y, y_pred)
    
    baseline_mae = mean_absolute_error(y, np.zeros_like(y))
    
    mae_delta = model_mae - baseline_mae
    
    return {
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_delta": mae_delta
    }

if __name__ == "__main__":
    raw_df = fetch_training_data()
    X_train, y_train, X_holdout, y_holdout = split_train_holdout(raw_df)
    
    candidate_model = train_candidate(X_train, y_train)
    
    model_training_metrics = evaluate_model(candidate_model, X_train, y_train)
    model_holdout_metrics = evaluate_model(candidate_model, X_holdout, y_holdout)
    
    print(f"training metrics: {model_training_metrics}")
    print(f"holdout metrics: {model_holdout_metrics}")