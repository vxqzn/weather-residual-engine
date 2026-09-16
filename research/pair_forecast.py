import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url_historical = "https://archive-api.open-meteo.com/v1/archive"
url_forecast = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
	"latitude": 45.7537,
	"longitude": 21.2257,
    "start_date": "2026-07-17",
    "end_date": "2026-09-14",
	"daily": "temperature_2m_max",
	"timezone": "GMT",
}
responses_historical = openmeteo.weather_api(url_historical, params = params)

response_historical = responses_historical[0]

daily_historical = response_historical.Daily()
daily_historical_temperature_2m_max = daily_historical.Variables(0).ValuesAsNumpy()

daily_data_historical = {
    "date": pd.date_range(
        start = pd.to_datetime(daily_historical.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(daily_historical.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily_historical.Interval()),
        inclusive = "left"
    ),
    "temp historical": daily_historical_temperature_2m_max
}
daily_dataframe_historical = pd.DataFrame(data = daily_data_historical)

responses_forecast = openmeteo.weather_api(url_forecast, params = params)

response_forecast = responses_forecast[0]

daily_forecast = response_forecast.Daily()
daily_forecast_temperature_2m_max = daily_forecast.Variables(0).ValuesAsNumpy()

daily_data_forecast = {
    "date": pd.date_range(
        start = pd.to_datetime(daily_forecast.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(daily_forecast.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily_forecast.Interval()),
        inclusive = "left"
    ),
    "temp forecast": daily_forecast_temperature_2m_max
}
daily_dataframe_forecast = pd.DataFrame(data = daily_data_forecast)

daily_dataframe = pd.merge(daily_dataframe_historical, daily_dataframe_forecast, on = "date", how = "outer")

daily_dataframe["forecast error"] = daily_dataframe["temp historical"] - daily_dataframe["temp forecast"]
baseline_mae = daily_dataframe["forecast error"].abs().mean()

print(f"Coordinates: {response_historical.Latitude()}°N {response_historical.Longitude()}°E")
print(f"Elevation: {response_historical.Elevation()} m asl")
print(f"Timezone: {response_historical.Timezone()}{response_historical.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response_historical.UtcOffsetSeconds()}s")

print("\nDaily data\n", daily_dataframe.head(10))
print(f"\nBaseline MAE: {baseline_mae:.2f}°C")