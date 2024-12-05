from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import math
import numpy as np
from pathlib import Path

@dataclass
class WeatherData:
    date: datetime
    tmin: float
    tmax: float
    tmean: float
    precip: float
    et0: float = 0.0

class Weather:
    """Unified weather module for soil-water-balance modeling"""
    
    def __init__(self, grid_shape: tuple):
        # Constants
        self.KRS = 0.0023  # Hargreaves constant
        self.GSC = 0.0820  # Solar constant (MJ/m²/min)
        self.NEAR_ZERO = 1.0e-9
        
        # Array initialization
        self.grid_shape = grid_shape
        self.tmin = np.array([])
        self.tmax = np.array([])
        self.precip = np.array([])
        self.weather_dates = []
        self.actual_et = np.zeros(grid_shape, dtype=np.float32)
        self.date_of_last_retrieval = None
        
        # State variables
        self._current_data: Optional[WeatherData] = None
        self.lapse_rate = -0.0065
        
    def initialize(self, data_file: Path) -> None:
        """Initialize weather data from file"""
        with open(data_file) as f:
            next(f)
            for line in f:
                date_str, tmin, tmax, precip = line.strip().split(',')
                self.tmin = np.append(self.tmin, float(tmin))
                self.tmax = np.append(self.tmax, float(tmax))
                self.precip = np.append(self.precip, float(precip))
                self.weather_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))

    def get_data_for_date(self, date: datetime, elevation: np.ndarray) -> WeatherData:
        """Get elevation-adjusted temperature and precipitation data"""
        try:
            idx = self.weather_dates.index(date)
        except ValueError:
            raise ValueError(f"No weather data available for {date}")
            
        # Apply elevation adjustments
        adj_tmin = self.tmin[idx] + self.lapse_rate * elevation
        adj_tmax = self.tmax[idx] + self.lapse_rate * elevation
        
        self._current_data = WeatherData(
            date=date,
            tmin=adj_tmin,
            tmax=adj_tmax,
            tmean=(adj_tmin + adj_tmax) / 2,
            precip=self.precip[idx]
        )
        return self._current_data

    def calculate_solar_parameters(self, day_of_year: int) -> Tuple[float, float, float]:
        """Calculate solar declination and related parameters"""
        delta = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)
        dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
        return delta, dr, self.GSC

    def calculate_extraterrestrial_radiation(self, latitude: float, day_of_year: int) -> float:
        """Calculate extraterrestrial radiation (Ra)"""
        phi = math.radians(latitude)
        delta, dr, gsc = self.calculate_solar_parameters(day_of_year)
        omega_s = math.acos(-math.tan(phi) * math.tan(delta))
        
        ra = (24 * 60 / math.pi) * gsc * dr * (
            omega_s * math.sin(phi) * math.sin(delta) +
            math.cos(phi) * math.cos(delta) * math.sin(omega_s)
        )
        return ra

    def calculate_et0(self, latitude: float) -> float:
        """Calculate reference ET using Hargreaves-Samani method"""
        if not self._current_data:
            raise ValueError("No current weather data loaded")
            
        doy = self._current_data.date.timetuple().tm_yday
        ra = self.calculate_extraterrestrial_radiation(latitude, doy)
        
        et0 = self.KRS * ra * (self._current_data.tmean + 17.8) * \
              (self._current_data.tmax - self._current_data.tmin)**0.5
              
        self._current_data.et0 = et0
        return et0

    def calculate_actual_et(self, soil_storage: np.ndarray,
                          soil_storage_max: float,
                          infiltration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate actual ET using Thornthwaite-Mather method"""
        if not self._current_data:
            raise ValueError("No current weather data loaded")
            
        soil_storage = np.array(soil_storage, dtype=np.float32)
        infiltration = np.array(infiltration, dtype=np.float32)
        potential_et = np.full_like(soil_storage, self._current_data.et0)
        
        actual_et = np.zeros_like(soil_storage, dtype=np.float32)
        
        # Calculate actual ET
        mask = potential_et <= (soil_storage + infiltration)
        actual_et[mask] = potential_et[mask]
        actual_et[~mask] = soil_storage[~mask] + infiltration[~mask]
        
        # Update soil storage
        soil_storage += infiltration
        soil_storage -= actual_et
        soil_storage = np.minimum(soil_storage, soil_storage_max)
        soil_storage[soil_storage < self.NEAR_ZERO] = 0.0
        
        if actual_et.shape == self.grid_shape:
            self.actual_et = actual_et
            self.date_of_last_retrieval = self._current_data.date
            
        return actual_et, soil_storage

    def calculate_vapor_pressure(self, temperature: float) -> float:
        """Calculate saturation vapor pressure"""
        return 0.6108 * math.exp((17.27 * temperature) / (temperature + 237.3))

    def get_summary(self) -> Dict:
        """Get weather data summary statistics"""
        return {
            "Total Days": len(self.weather_dates),
            "Average Tmin": np.mean(self.tmin) if len(self.tmin) > 0 else None,
            "Average Tmax": np.mean(self.tmax) if len(self.tmax) > 0 else None,
            "Total Precipitation": np.sum(self.precip) if len(self.precip) > 0 else None,
        }

    def get_et_value(self, row: int, col: int) -> float:
        """Get ET value for specific grid cell"""
        if self.actual_et is None:
            raise ValueError("ET grid not initialized")
        return self.actual_et[row, col]
