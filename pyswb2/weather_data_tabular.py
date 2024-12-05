
import numpy as np
from datetime import datetime

class WeatherDataTabular:
    def __init__(self):
        """Initialize weather data storage."""
        self.tmin = np.array([])  # Minimum temperatures
        self.tmax = np.array([])  # Maximum temperatures
        self.precip = np.array([])  # Precipitation values
        self.weather_date = []  # Dates associated with weather data
        self.date_index = 1  # Current index for processing

    def initialize(self, tmin, tmax, precip, dates):
        """Initialize weather data arrays and dates."""
        self.tmin = np.array(tmin, dtype=float)
        self.tmax = np.array(tmax, dtype=float)
        self.precip = np.array(precip, dtype=float)
        self.weather_date = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        self.date_index = 1

    def add_weather_data(self, tmin, tmax, precip, date):
        """Add a single weather data entry."""
        self.tmin = np.append(self.tmin, tmin)
        self.tmax = np.append(self.tmax, tmax)
        self.precip = np.append(self.precip, precip)
        self.weather_date.append(datetime.strptime(date, "%Y-%m-%d"))

    def get_summary(self):
        """Get a summary of the weather data."""
        return {
            "Total Days": len(self.weather_date),
            "Average Tmin": np.mean(self.tmin) if len(self.tmin) > 0 else None,
            "Average Tmax": np.mean(self.tmax) if len(self.tmax) > 0 else None,
            "Total Precipitation": np.sum(self.precip) if len(self.precip) > 0 else None,
        }


# Example Usage
if __name__ == "__main__":
    weather_data = WeatherDataTabular()

    # Initialize with some sample data
    weather_data.initialize(
        tmin=[-5.0, 0.0, 3.0],
        tmax=[5.0, 10.0, 15.0],
        precip=[0.1, 0.0, 0.5],
        dates=["2023-01-01", "2023-01-02", "2023-01-03"],
    )

    # Add additional data
    weather_data.add_weather_data(-2.0, 8.0, 0.2, "2023-01-04")

    # Print summary
    print("Weather Data Summary:")
    print(weather_data.get_summary())
