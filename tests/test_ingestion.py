import pytest
from pydantic import ValidationError
from datetime import date as dt_date, timedelta

from twre.db.session import get_connection
from twre.schemas.weather import ObservationRecord
from twre.ingestion.worker import ingest_observations, ingest_forecasts, parse_observations

def test_ingestion_idempotency():
    for _ in range(2):
        ingest_observations("2026-01-01", "2026-01-07")
        ingest_forecasts("2026-01-01", "2026-01-07")

    with get_connection() as conn:
        with conn.cursor() as cur:
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

    assert len(records) == 2
    assert records[0].date == dt_date(2026, 1, 1)
    assert records[1].date == dt_date(2026, 1, 3)