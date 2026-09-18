import openmeteo_requests
import requests
from datetime import date as dt_date, datetime, timedelta, timezone
from retry_requests import retry
from pydantic import ValidationError

from twre.db.session import get_connection
from twre.schemas.weather import ObservationRecord, ForecastRecord

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LATITUDE =  45.7537
LONGITUDE = 21.2257

UPSERT_OBSERVATION_SQL = """ 
    INSERT INTO weather_observations (
        date, temp_max, precipitation_sum, wind_speed_max, relative_humidity_mean
    ) VALUES (
        %(date)s, %(temp_max)s, %(precipitation_sum)s, %(wind_speed_max)s, %(relative_humidity_mean)s
    )
    ON CONFLICT (date) DO UPDATE SET
        temp_max = EXCLUDED.temp_max,
        precipitation_sum = EXCLUDED.precipitation_sum,
        wind_speed_max = EXCLUDED.wind_speed_max,
        relative_humidity_mean = EXCLUDED.relative_humidity_mean;
"""

UPSERT_FORECAST_SQL = """
    INSERT INTO forecast_predictions (
        target_date, issued_date, forecasted_temp_max, forecasted_precipitation_sum, forecasted_wind_speed_max, forecasted_relative_humidity_mean
    ) VALUES (
        %(target_date)s, %(issued_date)s, %(forecasted_temp_max)s, %(forecasted_precipitation_sum)s, %(forecasted_wind_speed_max)s, %(forecasted_relative_humidity_mean)s
    )
    ON CONFLICT (target_date, issued_date) DO UPDATE SET
        forecasted_temp_max = EXCLUDED.forecasted_temp_max,
        forecasted_precipitation_sum = EXCLUDED.forecasted_precipitation_sum,
        forecasted_wind_speed_max = EXCLUDED.forecasted_wind_speed_max,
        forecasted_relative_humidity_mean = EXCLUDED.forecasted_relative_humidity_mean;
"""

COMPUTE_ERRORS_SQL = """
    INSERT INTO realized_errors (
        target_date, issued_date, error_temp_max, absolute_error_temp_max
    )
    SELECT
        f.target_date,
        f.issued_date,
        o.temp_max - f.forecasted_temp_max AS error_temp_max,
        ABS(o.temp_max - f.forecasted_temp_max) AS absolute_error_temp_max
    FROM forecast_predictions f
    JOIN weather_observations o ON f.target_date = o.date
    ON CONFLICT (target_date, issued_date) DO UPDATE SET
        error_temp_max = EXCLUDED.error_temp_max,
        absolute_error_temp_max = EXCLUDED.absolute_error_temp_max;
"""

def get_client() -> openmeteo_requests.Client:
    session = retry(requests.Session(), retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)

def parse_observations(
        start_date: dt_date,
        temps: list[float],
        precips: list[float],
        winds: list[float],
        humidities: list[float],
    ) -> list[ObservationRecord]:
        records: list[ObservationRecord] = []
        for i in range(len(temps)):
            row_date = start_date + timedelta(days=i)
            try:
                record = ObservationRecord(
                    date=row_date,
                    temp_max=float(temps[i]),
                    precipitation_sum=float(precips[i]),
                    wind_speed_max=float(winds[i]),
                    relative_humidity_mean=float(humidities[i]),
                )
                records.append(record)
            except ValidationError as err:
                print(f"Skipping corrupted observation for {row_date}: {err}")
                continue
        return records

def ingest_observations(start_date: str, end_date: str) -> int:
    client = get_client()
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean"],
        "timezone": "GMT",
    }
    responses = client.weather_api(ARCHIVE_API_URL, params=params)
    daily = responses[0].Daily()

    start_dt = datetime.fromtimestamp(daily.Time(), tz=timezone.utc).date()
    temps = daily.Variables(0).ValuesAsNumpy()
    precips = daily.Variables(1).ValuesAsNumpy()
    winds = daily.Variables(2).ValuesAsNumpy()
    humidities = daily.Variables(3).ValuesAsNumpy()

    records = parse_observations(start_dt, temps, precips, winds, humidities)
    
    payloads = [r.model_dump() for r in records]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_OBSERVATION_SQL, payloads)

    return len(payloads)

def ingest_forecasts(start_date: str, end_date: str) -> int:
    client = get_client()
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean"],
        "timezone": "GMT",
    }
    responses = client.weather_api(FORECAST_API_URL, params=params)
    daily = responses[0].Daily()

    start_dt = datetime.fromtimestamp(daily.Time(), tz=timezone.utc).date()
    temps = daily.Variables(0).ValuesAsNumpy()
    precips = daily.Variables(1).ValuesAsNumpy()
    winds = daily.Variables(2).ValuesAsNumpy()
    humidities = daily.Variables(3).ValuesAsNumpy()

    records: list[ForecastRecord] = []
    for i in range(len(temps)):
        row_date = start_dt + timedelta(days=i)

        target_dt = start_dt + timedelta(days=i)
        issued_dt = target_dt - timedelta(days=1)
        try:
            record = ForecastRecord(
                target_date=target_dt,
                issued_date=issued_dt,
                forecasted_temp_max=float(temps[i]),
                forecasted_precipitation_sum=float(precips[i]),
                forecasted_wind_speed_max=float(winds[i]),
                forecasted_relative_humidity_mean=float(humidities[i])
            )
            records.append(record)
        except ValidationError as err:
            print(f"Skipping corrupted forecast for {row_date}: {err}")
            continue

    payloads = [r.model_dump() for r in records]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_FORECAST_SQL, payloads)

    return len(payloads)

def compute_realized_errors() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(COMPUTE_ERRORS_SQL)
            return cur.rowcount

if __name__ == "__main__":
    # Test 7d slice
    count = ingest_observations("2026-01-01", "2026-01-07")
    count2 = ingest_forecasts("2026-01-01", "2026-01-07")
    count3 = compute_realized_errors()
    print(f"upserted {count} observation records")
    print(f"\nupserted {count2} forecast records")
    print(f"\ncomputed {count3} error records")