import pytest
from pydantic import ValidationError
from datetime import date as dt_date

from twre.schemas.weather import ObservationRecord, ForecastRecord

def test_valid_observation_instantiation():
    record = ObservationRecord(
        date=dt_date(2026, 9, 18),
        temp_max=26.0,
        precipitation_sum=10.0,
        wind_speed_max=20.0,
        relative_humidity_mean=80.0,
    )
    assert record.date == dt_date(2026, 9, 18)
    assert record.temp_max == 26.0
    assert record.precipitation_sum == 10.0
    assert record.wind_speed_max == 20.0
    assert record.relative_humidity_mean == 80.0

def test_observation_bound_violations():
    with pytest.raises(ValidationError):
        ObservationRecord(
            date=dt_date(2026, 9, 18),
            temp_max=999.0,
            precipitation_sum=10.0,
            wind_speed_max=20.0,
            relative_humidity_mean=80.0,
        )
    with pytest.raises(ValidationError):
            ObservationRecord(
            date=dt_date(2026, 9, 18),
            temp_max=-999.0,
            precipitation_sum=10.0,
            wind_speed_max=20.0,
            relative_humidity_mean=80.0,
            )
    with pytest.raises(ValidationError):
        ObservationRecord(
            date=dt_date(2026, 9, 18),
            temp_max=30.0,
            precipitation_sum=-999.0,
            wind_speed_max=20.0,
            relative_humidity_mean=80.0,
        )
    with pytest.raises(ValidationError):
        ObservationRecord(
            date=dt_date(2026, 9, 18),
            temp_max=30.0,
            precipitation_sum=10.0,
            wind_speed_max=-999.0,
            relative_humidity_mean=80.0,
        )
    with pytest.raises(ValidationError):
        ObservationRecord(
            date=dt_date(2026, 9, 18),
            temp_max=30.0,
            precipitation_sum=10.0,
            wind_speed_max=20.0,
            relative_humidity_mean=999.0,
        )
    with pytest.raises(ValidationError):
        ObservationRecord(
        date=dt_date(2026, 9, 18),
        temp_max=30.0,
        precipitation_sum=10.0,
        wind_speed_max=20.0,
        relative_humidity_mean=-999.0,
        )
def test_valid_forecast_instantiation():
    record = ForecastRecord(
        target_date=dt_date(2026, 9, 16),
        issued_date=dt_date(2026, 9, 15),
        forecasted_temp_max=30.0,
        forecasted_precipitation_sum=10.0,
        forecasted_wind_speed_max=20.0,
        forecasted_relative_humidity_mean=80.0,
    )
    assert record.target_date == dt_date(2026, 9, 16)
    assert record.issued_date == dt_date(2026, 9, 15)
    assert record.forecasted_temp_max == 30.0
    assert record.forecasted_precipitation_sum == 10.0
    assert record.forecasted_wind_speed_max == 20.0
    assert record.forecasted_relative_humidity_mean == 80.0

def test_forecast_temporal_order_violation():
    with pytest.raises(ValidationError):
        record = ForecastRecord(
            target_date=dt_date(2026, 9, 15),
            issued_date=dt_date(2026, 9, 16),
            forecasted_temp_max=30.0,
            forecasted_precipitation_sum=10.0,
            forecasted_wind_speed_max=20.0,
            forecasted_relative_humidity_mean=80.0,
        )

def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        ObservationRecord(
            date=dt_date(2026, 9, 18),
            temp_max=30.0,
            precipitation_sum=10.0,
            wind_speed_max=20.0,
            relative_humidity_mean=80.0,
            something="something extra"
        )
    with pytest.raises(ValidationError):
        ForecastRecord(
            target_date=dt_date(2026, 9, 16),
            issued_date=dt_date(2026, 9, 15),
            forecasted_temp_max=30.0,
            forecasted_precipitation_sum=10.0,
            forecasted_wind_speed_max=20.0,
            forecasted_relative_humidity_mean=80.0,
            something="something extra"
        )

def test_records_are_frozen():
    record = ObservationRecord(
        date=dt_date(2026, 9, 18),
        temp_max=30.0,
        precipitation_sum=10.0,
        wind_speed_max=20.0,
        relative_humidity_mean=80.0,
    )
    with pytest.raises(ValidationError):
        record.temp_max = 35.0