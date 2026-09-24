import pytest
import numpy as np
from unittest.mock import MagicMock
from datetime import date as dt_date, datetime, timezone

from twre.db.session import get_connection
from twre.ingestion.worker import ingest_observations, ingest_forecasts, parse_observations, parse_forecasts, compute_realized_errors

@pytest.fixture
def mock_openmeteo_client():
    client = MagicMock()
    response = MagicMock()
    daily = MagicMock()
    
    daily.Time.return_value = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    
    data_map = {
        0: np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 21.5]),   # temps
        1: np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),          # precips
        2: np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),   # winds
        3: np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0]),   # humidities
    }
    
    def get_variable(index):
        var = MagicMock()
        var.ValuesAsNumpy.return_value = data_map[index]
        return var
    
    daily.Variables.side_effect = get_variable
    response.Daily.return_value = daily
    client.weather_api.return_value = [response]
    return client

@pytest.fixture
def db_conn():
    with get_connection() as conn:
        with conn.transaction(force_rollback=True):    
            yield conn

def test_ingestion_idempotency(db_conn, mock_openmeteo_client):
    for _ in range(2):
        ingest_observations(conn=db_conn, client=mock_openmeteo_client, start_date="2026-01-01", end_date="2026-01-07")
        ingest_forecasts(conn=db_conn, client=mock_openmeteo_client, start_date="2026-01-01", end_date="2026-01-07")

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM weather_observations WHERE date BETWEEN '2026-01-01' AND '2026-01-07';")
        (obs_count,) = cur.fetchone()
        assert obs_count == 7

        cur.execute("SELECT COUNT(*) FROM forecast_predictions WHERE target_date BETWEEN '2026-01-01' AND '2026-01-07';")
        (fc_count,) = cur.fetchone()
        assert fc_count == 7

def test_quarantine_rejects_corruptions():
    start = dt_date(2026, 1, 1)
    temps = [20.0, -999.0, 22.0]
    precips = [0.0, 0.0, 0.0]
    winds = [10.0, 10.0, 10.0]
    humidities = [50.0, 50.0, 50.0]

    records = parse_observations(start, temps, precips, winds, humidities)
    records2 = parse_forecasts(start, temps, precips, winds, humidities)

    assert len(records) == 2
    assert records[0].date == dt_date(2026, 1, 1)
    assert records[1].date == dt_date(2026, 1, 3)

    assert len(records2) == 2
    assert records2[0].target_date == dt_date(2026, 1, 1)
    assert records2[1].target_date == dt_date(2026, 1, 3)
    
def test_compute_realized_errors(db_conn, mock_openmeteo_client):
    ingest_observations(conn=db_conn, client=mock_openmeteo_client, start_date="2026-01-01", end_date="2026-01-07")
    ingest_forecasts(conn=db_conn, client=mock_openmeteo_client, start_date="2026-01-01", end_date="2026-01-07")

    count = compute_realized_errors(conn=db_conn, start_date=dt_date(2026, 1, 1), end_date=dt_date(2026, 1, 7))
    assert isinstance(count, int)
    assert count == 7

    with db_conn.cursor() as cur:
        cur.execute("""
                SELECT r.error_temp_max, r.absolute_error_temp_max, o.temp_max, f.forecasted_temp_max
                FROM realized_errors r
                JOIN weather_observations o ON r.target_date = o.date
                JOIN forecast_predictions f ON r.target_date = f.target_date AND r.issued_date = f.issued_date
                WHERE r.target_date BETWEEN '2026-01-01' AND '2026-01-07';        
            """)
        rows = cur.fetchall()
        assert len(rows) == 7
        for err, abs_err, obs_temp, fc_temp in rows:
            assert err == obs_temp - fc_temp
            assert abs_err == abs(err)