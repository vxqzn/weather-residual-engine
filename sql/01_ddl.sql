CREATE TABLE IF NOT EXISTS weather_observations (
    date DATE PRIMARY KEY,
    temp_max DOUBLE PRECISION NOT NULL,
    precipitation_sum DOUBLE PRECISION NOT NULL,
    wind_speed_max DOUBLE PRECISION NOT NULL,
    relative_humidity_mean DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT check_temp_bounds CHECK (temp_max BETWEEN -50 AND 60),
    CONSTRAINT check_precipitation_positive CHECK (precipitation_sum >= 0),
    CONSTRAINT check_wind_speed_positive CHECK (wind_speed_max >= 0),
    CONSTRAINT check_humidity_pct CHECK (relative_humidity_mean BETWEEN 0 AND 100)
);

CREATE TABLE IF NOT EXISTS forecast_predictions (
    target_date DATE,
    issued_date DATE,
    forecasted_temp_max DOUBLE PRECISION NOT NULL,
    forecasted_precipitation_sum DOUBLE PRECISION NOT NULL,
    forecasted_wind_speed_max DOUBLE PRECISION NOT NULL,
    forecasted_relative_humidity_mean DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (target_date, issued_date),
    CONSTRAINT check_temp_bounds CHECK (forecasted_temp_max BETWEEN -50 AND 60),
    CONSTRAINT check_precipitation_positive CHECK (forecasted_precipitation_sum >= 0),
    CONSTRAINT check_wind_speed_positive CHECK (forecasted_wind_speed_max >= 0),
    CONSTRAINT check_humidity_pct CHECK (forecasted_relative_humidity_mean BETWEEN 0 AND 100)
);

CREATE TABLE IF NOT EXISTS realized_errors (
    target_date DATE NOT NULL,
    issued_date DATE NOT NULL,
    error_temp_max DOUBLE PRECISION NOT NULL,
    absolute_error_temp_max DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    FOREIGN KEY (target_date, issued_date) REFERENCES forecast_predictions(target_date, issued_date) ON DELETE CASCADE,
    FOREIGN KEY (target_date) REFERENCES weather_observations(date) ON DELETE CASCADE,
    PRIMARY KEY (target_date, issued_date),

    CONSTRAINT check_abs_Error_non_negative CHECK (absolute_error_temp_max >= 0)
);