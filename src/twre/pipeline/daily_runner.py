from datetime import date, timedelta
from typing import Any

from twre.db.session import resolve_connection
from twre.ingestion.worker import (
    ingest_observations,
    ingest_forecasts,
    compute_realized_errors
)
from twre.features.pipeline import fetch_training_data
from twre.models.train import split_train_holdout, train_candidate
from twre.models.gate import evaluate_and_promote

def run_daily_pipeline(lookback_days: int = 14, holdout_days: int = 90, conn=None) -> dict[str, Any]:
    with resolve_connection(conn) as active_conn:
        today = date.today()
        start_date = (today - timedelta(days=lookback_days)).isoformat()
        end_date = today.isoformat()
        expected_days = lookback_days + 1
        
        print(f"running daily pipeline for {start_date} to {end_date}...")
        
        ingest_observations(conn=active_conn, start_date=start_date, end_date=end_date)
        ingest_forecasts(conn=active_conn, start_date=start_date, end_date=end_date)
        compute_realized_errors(conn=active_conn, start_date=today - timedelta(days=lookback_days), end_date=today)
        
        with active_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM weather_observations
                WHERE date BETWEEN %(start_date)s AND %(end_date)s;
            """, { "start_date": start_date, "end_date": end_date })
            obs_count = cur.fetchone()[0]
        
            cur.execute("""
                SELECT COUNT(*) 
                FROM forecast_predictions
                WHERE target_date BETWEEN %(start_date)s AND %(end_date)s;
            """, { "start_date": start_date, "end_date": end_date })
            fc_count = cur.fetchone()[0]
        
            cur.execute("""
                SELECT COUNT(*)
                FROM realized_errors
                WHERE target_date BETWEEN %(start_date)s AND %(end_date)s;
            """, { "start_date": start_date, "end_date": end_date })
            err_count = cur.fetchone()[0]
            
            assert obs_count == expected_days, f"observations mismatch: {expected_days} != {obs_count}"
            assert fc_count == expected_days, f"forecasts mismatch: {expected_days} != {fc_count}"
            assert err_count == expected_days, f"realized errors mismatch: {expected_days} != {err_count}" 
        
        print(f"found {obs_count} observations, {fc_count} forecasts, and {err_count} errors...")
        print("\ntraining and evaluating candidate model...")
        
        raw_df = fetch_training_data(conn=active_conn)
        X_train, y_train, X_holdout, y_holdout = split_train_holdout(raw_df, holdout_days=holdout_days)
        candidate = train_candidate(X_train, y_train)
        
        metadata, is_promoted = evaluate_and_promote(
            candidate,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
        )
        
        print(f"candidate model evaluation results: {metadata}")
        print(f"candidate model promoted: {is_promoted}")
            
        return {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "observations_upserted": obs_count,
            "forecasts_upserted": fc_count,
            "errors_computed": err_count,
            "candidate_model_id": metadata.model_id,
            "is_promoted": is_promoted,
            "holdout_model_mae": metadata.holdout_model_mae,
            "promotion_reason": metadata.promotion_reason,
            "train_window": f"{X_train.index.min().date()} to {X_train.index.max().date()}",
            "holdout_window": f"{X_holdout.index.min().date()} to {X_holdout.index.max().date()}" 
        }
        
if __name__ == "__main__":
    result = run_daily_pipeline()
    print(f"daily pipeline completed successfully: {result}")