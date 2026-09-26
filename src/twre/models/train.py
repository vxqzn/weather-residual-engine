import pandas as pd
from sklearn.linear_model import LinearRegression

from twre.features.pipeline import build_feature_matrix, fetch_training_data

FEATURE_COLUMNS = (
    "error_lag_1",
    "rolling_bias_7",
    "forecasted_temp_max"
)
TARGET_COLUMN = "error_temp_max"

def split_train_holdout(df: pd.DataFrame, holdout_days: int = 90, cutoff_date: str = "2024-01-01", min_train_days: int = 365) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = build_feature_matrix(df)    
    max_date = df.index.max()
    split_boundary = max_date - pd.Timedelta(days=holdout_days)

    train_df = df.loc[cutoff_date:split_boundary]
    holdout_df = df.loc[split_boundary + pd.Timedelta(days=1):max_date]
    
    X_train = train_df[list(FEATURE_COLUMNS)]
    y_train = train_df[TARGET_COLUMN]
    X_holdout = holdout_df[list(FEATURE_COLUMNS)]
    y_holdout = holdout_df[TARGET_COLUMN]
    
    assert len(train_df) >= min_train_days, f"expected at least {min_train_days} training rows, got {len(train_df)}"
    assert len(holdout_df) == holdout_days, f"expected {holdout_days} holdout rows, got {len(holdout_df)}"
    assert train_df.index.max() < holdout_df.index.min(), "train and holdout sets are overlapping"
    
    assert X_train.notna().all().all(), "training features contain NaN values"
    assert X_holdout.notna().all().all(), "holdout features contain NaN values"
    assert y_train.notna().all().all(), "training target contains NaN values"
    assert y_holdout.notna().all().all(), "holdout target contains NaN values"

    return X_train, y_train, X_holdout, y_holdout

def train_candidate(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

if __name__ == "__main__":
    raw_df = fetch_training_data()
    X_train, y_train, X_holdout, y_holdout = split_train_holdout(raw_df)
    
    model = train_candidate(X_train, y_train)
    
    print(f"model coefficients: {model.coef_}")
    print(f"model intercept: {model.intercept_}")