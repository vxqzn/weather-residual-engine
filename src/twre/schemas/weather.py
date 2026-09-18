from datetime import date as dt_date
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ObservationRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True
        )

    date: dt_date = Field(description="Date of observation")
    temp_max: float = Field(ge=-50, le=60, description="Maximum temperature in Celsius")
    precipitation_sum: float = Field(ge=0, description="Total precipitation in mm")
    wind_speed_max: float = Field(ge=0, description="Maximum wind speed in m/s")
    relative_humidity_mean: float = Field(ge=0, le=100, description="Mean relative humidity in %")
    
class ForecastRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True
        )

    target_date: dt_date = Field(description="Date for which the forecast is made")
    issued_date: dt_date = Field(description="Date when the forecast was issued")
    forecasted_temp_max: float = Field(ge=-50, le=60, description="Maximum temperature in Celsius")
    forecasted_precipitation_sum: float = Field(ge=0, description="Total precipitation in mm")
    forecasted_wind_speed_max: float = Field(ge=0, description="Maximum wind speed in m/s")
    forecasted_relative_humidity_mean: float = Field(ge=0, le=100, description="Mean relative humidity in %")

    @model_validator(mode="after")
    def check_temporal_consistency(self):
        if self.issued_date > self.target_date:
            raise ValueError("Issued date cannot be after target date")
        return self