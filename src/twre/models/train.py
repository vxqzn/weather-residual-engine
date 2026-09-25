import pandas as pd
from sklearn.linear_model import LinearRegression

from twre.features.pipeline import build_feature_matrix, fetch_training_data

FEATURE_COLUMNS = (
    "error_lag_1",
    "rolling_bias_7",
    "forecasted_temp_max"
)
TARGET_COLUMN = "error_temp_max"

def split_train_holdout(df: pd.DataFrame):
    df = build_feature_matrix(df)
    train_df = df.loc["2024-01-01":"2024-12-31"]
    holdout_df = df.loc["2025-01-01":"2025-12-31"]
    
    X_train = train_df[list(FEATURE_COLUMNS)]
    y_train = train_df[TARGET_COLUMN]
    X_holdout = holdout_df[list(FEATURE_COLUMNS)]
    y_holdout = holdout_df[TARGET_COLUMN]
    
    assert len(train_df) == 366, f"expected 366 training rows, got {len(train_df)}"
    assert len(holdout_df) == 365, f"expected 365 holdout rows, got {len(holdout_df)}"
    
    assert X_train.notna().all().all(), "training features contain NaN values"
    assert X_holdout.notna().all().all(), "holdout features contain NaN values"
    assert y_train.notna().all().all(), "training target contains NaN values"
    assert y_holdout.notna().all().all(), "holdout target contains NaN values"

    return X_train, y_train, X_holdout, y_holdout

def train_candidate(X_train: pd.DataFrame, y_train: pd.DataFrame, alpha: float = 1.0) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

if __name__ == "__main__":
    raw_df = fetch_training_data()
    X_train, y_train, X_holdout, y_holdout = split_train_holdout(raw_df)
    
    model = train_candidate(X_train, y_train)
    
    print(f"model coefficients: {model.coef_}")
    print(f"model intercept: {model.intercept_}")