import openmeteo_requests

import pandas as pd
import requests_cache

from typing import Any
from retry_requests import retry
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url_historical = "https://archive-api.open-meteo.com/v1/archive"
url_forecast = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
	"latitude": 45.7537,
	"longitude": 21.2257,
    "start_date": "2024-12-18",
    "end_date": "2026-01-01",
	"daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean"],
	"timezone": "GMT",
}

responses_historical = openmeteo.weather_api(url_historical, params = params)
responses_forecast = openmeteo.weather_api(url_forecast, params = params)

response_historical = responses_historical[0]
response_forecast = responses_forecast[0]

daily_historical = response_historical.Daily()
daily_historical_temperature_2m_max = daily_historical.Variables(0).ValuesAsNumpy()
daily_historical_precipitation_sum = daily_historical.Variables(1).ValuesAsNumpy()
daily_historical_wind_speed_10m_max = daily_historical.Variables(2).ValuesAsNumpy()
daily_historical_relative_humidity_2m_mean = daily_historical.Variables(3).ValuesAsNumpy()

daily_forecast = response_forecast.Daily()
daily_forecast_temperature_2m_max = daily_forecast.Variables(0).ValuesAsNumpy()
daily_forecast_precipitation_sum = daily_forecast.Variables(1).ValuesAsNumpy()
daily_forecast_wind_speed_10m_max = daily_forecast.Variables(2).ValuesAsNumpy()
daily_forecast_relative_humidity_2m_mean = daily_forecast.Variables(3).ValuesAsNumpy()

daily_data_historical: dict[str, Any] = {
    "date": pd.date_range(
        start = pd.to_datetime(daily_historical.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(daily_historical.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily_historical.Interval()),
        inclusive = "left"
    )
}

daily_data_forecast: dict[str, Any] = {
    "date": pd.date_range(
        start = pd.to_datetime(daily_forecast.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(daily_forecast.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily_forecast.Interval()),
        inclusive = "left"
    )
}

daily_data_historical["temp historical"] = daily_historical_temperature_2m_max
daily_data_historical["precipitation_historical"] = daily_historical_precipitation_sum
daily_data_historical["wind_speed_historical"] = daily_historical_wind_speed_10m_max
daily_data_historical["relative_humidity_historical"] = daily_historical_relative_humidity_2m_mean

daily_data_forecast["temp forecast"] = daily_forecast_temperature_2m_max
daily_data_forecast["precipitation_forecast"] = daily_forecast_precipitation_sum
daily_data_forecast["wind_speed_forecast"] = daily_forecast_wind_speed_10m_max
daily_data_forecast["relative_humidity_forecast"] = daily_forecast_relative_humidity_2m_mean

daily_dataframe_historical = pd.DataFrame(data = daily_data_historical)
daily_dataframe_forecast = pd.DataFrame(data = daily_data_forecast)

daily_dataframe = pd.merge(daily_dataframe_historical, daily_dataframe_forecast, on = "date", how = "inner")

daily_dataframe["forecast error (temperature)"] = daily_dataframe["temp historical"] - daily_dataframe["temp forecast"]
daily_dataframe["rolling mean 7d (temperature)"] = daily_dataframe["forecast error (temperature)"].shift(1).rolling(7).mean()
daily_dataframe["error_lag_1d (temperature)"] = daily_dataframe["forecast error (temperature)"].shift(1)

data = daily_dataframe[daily_dataframe["date"] >= pd.to_datetime("2025-01-01", utc = True)].copy()

feat_columns = ["rolling mean 7d (temperature)", "error_lag_1d (temperature)"]
assert data[feat_columns].notna().all().all(), "warmup NaN values found"

baseline_mae = data["forecast error (temperature)"].abs().mean()

print(f"Coordinates: {response_historical.Latitude()}°N {response_historical.Longitude()}°E")
print(f"Elevation: {response_historical.Elevation()} m asl")
print(f"Timezone: {response_historical.Timezone()}{response_historical.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response_historical.UtcOffsetSeconds()}s")

print("\nDaily data\n", data)
print(f"\nBaseline MAE: {baseline_mae:.2f}°C")

feature_columns = [
    "error_lag_1d (temperature)",
    "rolling mean 7d (temperature)",
    "temp forecast"
]

split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

X_train = train_data[feature_columns]
y_train = train_data["forecast error (temperature)"]
X_test = test_data[feature_columns]
y_test = test_data["forecast error (temperature)"]

model = LinearRegression()

model.fit(X_train, y_train)

predicted_error = model.predict(X_test)

corrected_forecast = test_data["temp forecast"] + predicted_error

print(f"Mean Absolute Error (baseline): {mean_absolute_error(test_data["temp historical"], test_data["temp forecast"]):.4f}°C")
print(f"Mean Absolute Error (corrected): {mean_absolute_error(test_data["temp historical"], corrected_forecast):.4f}°C")