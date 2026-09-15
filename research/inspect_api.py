import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 45.7537,
	"longitude": 21.2257,
	"daily": ["temperature_2m_min", "temperature_2m_max", "apparent_temperature_min", "apparent_temperature_max", "weather_code", "precipitation_probability_max", "uv_index_max", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"],
	"timezone": "GMT",
	"start_date": "2026-09-01",
	"end_date": "2026-09-14",
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_temperature_2m_min = daily.Variables(0).ValuesAsNumpy()
daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
daily_apparent_temperature_min = daily.Variables(2).ValuesAsNumpy()
daily_apparent_temperature_max = daily.Variables(3).ValuesAsNumpy()
daily_weather_code = daily.Variables(4).ValuesAsNumpy()
daily_precipitation_probability_max = daily.Variables(5).ValuesAsNumpy()
daily_uv_index_max = daily.Variables(6).ValuesAsNumpy()
daily_wind_speed_10m_max = daily.Variables(7).ValuesAsNumpy()
daily_wind_gusts_10m_max = daily.Variables(8).ValuesAsNumpy()
daily_wind_direction_10m_dominant = daily.Variables(9).ValuesAsNumpy()

daily_data = {
	"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)
}

daily_data["temperature_2m_min"] = daily_temperature_2m_min
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
daily_data["weather_code"] = daily_weather_code
daily_data["precipitation_probability_max"] = daily_precipitation_probability_max
daily_data["uv_index_max"] = daily_uv_index_max
daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant

daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)
print("\nDaily\n", daily_data)