from contextlib import contextmanager
from twre.db.session import get_connection, resolve_connection
from twre.ingestion.worker import ingest_observations, ingest_forecasts, compute_realized_errors
from datetime import date

START_DATE = "2023-12-18"
END_DATE = "2025-12-31"
EXPECTED_ROWS = 745

def verify_ingestion(conn=None, expected_count: int = EXPECTED_ROWS) -> dict[str, int]:
    with resolve_connection(conn) as active_conn:
        with active_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM weather_observations")
            (obs_count,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM forecast_predictions")
            (fc_count,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM realized_errors")
            (re_count,) = cur.fetchone()
            
            assert obs_count == expected_count, f"observations mismatch: {expected_count} != {obs_count}"
            assert fc_count == expected_count, f"forecasts mismatch: {expected_count} != {fc_count}"
            assert re_count == expected_count, f"realized errors mismatch: {expected_count} != {re_count}"
            
            return {
                "observations": obs_count,
                "forecasts": fc_count,
                "realized_errors": re_count,
                "expected": expected_count
            }

def run_backfill(start_date: str = START_DATE, end_date: str = END_DATE) -> None:
    with get_connection() as conn:
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
        
        print (f"targeted backfill from {start_date} to {end_date}")
        ingest_observations(conn=conn, start_date=start_date, end_date=end_date)
        ingest_forecasts(conn=conn, start_date=start_date, end_date=end_date)
        compute_realized_errors(conn=conn, start_date=start_dt, end_date=end_dt)
    
        verify_ingestion(conn=conn, expected_count=(end_dt - start_dt).days + 1)
        
if __name__ == "__main__":
    run_backfill()
    print(f"backfill completed successfully for {START_DATE} to {END_DATE}")