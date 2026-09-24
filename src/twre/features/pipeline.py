import pandas as pd
from twre.ingestion.backfill import START_DATE, END_DATE
from twre.db.session import resolve_connection

def fetch_training_data(conn=None, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    with resolve_connection(conn) as active_conn:
        with active_conn.cursor() as cur:
            cur.execute("""
                SELECT
                    p.target_date,
                    p.forecasted_temp_max,
                    p.forecasted_precipitation_sum,
                    p.forecasted_wind_speed_max,
                    p.forecasted_relative_humidity_mean,
                    o.temp_max AS observed_temp_max,
                    e.error_temp_max
                FROM forecast_predictions p
                JOIN weather_observations o ON p.target_date = o.date
                JOIN realized_errors e ON p.target_date = e.target_date 
                    AND p.issued_date = e.issued_date
                WHERE p.target_date BETWEEN %(start_date)s AND %(end_date)s
                ORDER BY p.target_date
            """, 
            {"start_date": start_date, "end_date": end_date}
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
            return pd.DataFrame(rows, columns=columns)

def build_feature_matrix(
    df: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cutoff_date: str = "2024-01-01"
    ) -> pd.DataFrame:
    if df is None:
        df = fetch_training_data(start_date=start_date, end_date=end_date)
    df = df.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    
    df = df.set_index("target_date")
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.reindex(full_index)
    
    df["error_lag_1"] = df["error_temp_max"].shift(1)
    df["rolling_bias_7"] = df["error_temp_max"].shift(1).rolling(7).mean()
    
    feature_df = df.loc[cutoff_date:]
    assert feature_df.notna().all().all(), "feature matrix contains NaN values"
    
    return feature_df
    
if __name__ == "__main__":
    raw_df = fetch_training_data(start_date=START_DATE, end_date=END_DATE)
    features = build_feature_matrix(raw_df, start_date=START_DATE, end_date=END_DATE)
    
    print(f"raw shape: {raw_df.shape}")
    print(f"raw data:\n{raw_df.head(3)}")
    print("--------------------------------")
    print(f"\nfeatures shape: {features.shape}")
    print(f"features data:\n{features.head(3)}")