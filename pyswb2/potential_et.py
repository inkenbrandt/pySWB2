from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
import numpy as np
from numpy.typing import NDArray

@dataclass
class HargreavesSamaniParams:
    """Parameters for Hargreaves-Samani ET calculation"""
    et_slope: float = 0.0023    # Slope coefficient
    et_exponent: float = 0.5    # Temperature range exponent
    et_constant: float = 17.8   # Temperature constant

@dataclass
class JensenHaiseParams:
    """Parameters for Jensen-Haise ET calculation"""
    as_coef: float = 0.25      # Solar radiation coefficient a
    bs_coef: float = 0.50      # Solar radiation coefficient b
    ct_coef: float = 0.014     # Temperature coefficient
    tx_coef: float = -0.37     # Temperature constant

@dataclass
class GriddedETParams:
    """Parameters for gridded ET values"""
    et_zones: NDArray[np.int32]
    et_ratios: NDArray[np.float32]
    monthly_ratios: Dict[int, List[float]]  # Zone ID -> monthly ratios

class PotentialETCalculator:
    """Calculator for potential evapotranspiration using various methods"""
    
    def __init__(self, domain_size: int, method: Literal["hargreaves", "jensen_haise", "gridded"] = "hargreaves"):
        """Initialize ET calculator
        
        Args:
            domain_size: Number of cells in model domain
            method: ET calculation method to use
        """
        if method not in ["hargreaves", "jensen_haise", "gridded"]:
            raise ValueError("method must be one of: 'hargreaves', 'jensen_haise', 'gridded'")
            
        self.method = method
        self.domain_size = domain_size
        
        # Initialize arrays
        self.reference_et0 = np.zeros(domain_size, dtype=np.float32)
        self.latitude = np.zeros(domain_size, dtype=np.float32)
        
        # Parameters for different methods
        self.hargreaves_params = HargreavesSamaniParams()
        self.jensen_haise_params = JensenHaiseParams()
        self.gridded_params: Optional[GriddedETParams] = None
        
    def initialize_gridded(self, et_zones: NDArray[np.int32], 
                         monthly_ratios: Dict[int, List[float]]) -> None:
        """Initialize gridded ET parameters
        
        Args:
            et_zones: Array mapping cells to ET zones
            monthly_ratios: Dictionary mapping zone IDs to monthly ET ratios
        """
        if self.method != "gridded":
            raise ValueError("Can only initialize gridded parameters when using gridded method")
            
        # Initialize ratios array
        et_ratios = np.zeros(self.domain_size, dtype=np.float32)
        
        # Set initial ratios from first month
        for zone_id, ratios in monthly_ratios.items():
            mask = et_zones == zone_id
            et_ratios[mask] = ratios[0]  # January ratio
            
        self.gridded_params = GriddedETParams(
            et_zones=et_zones,
            et_ratios=et_ratios,
            monthly_ratios=monthly_ratios
        )
        
    def _calculate_solar_parameters(self, day_of_year: int,
                                  days_in_year: int = 365) -> tuple[float, float, float]:
        """Calculate solar radiation parameters
        
        Args:
            day_of_year: Day of year (1-365/366)
            days_in_year: Number of days in year
            
        Returns:
            Tuple of (relative distance, declination, sunset angle)
        """
        # Relative earth-sun distance
        dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / days_in_year)
        
        # Solar declination
        delta = 0.409 * np.sin(2.0 * np.pi * day_of_year / days_in_year - 1.39)
        
        # Sunset hour angle
        omega_s = np.arccos(-np.tan(self.latitude) * np.tan(delta))
        
        return dr, delta, omega_s
        
    def _calculate_extraterrestrial_radiation(self, dr: float, delta: float, 
                                            omega_s: float) -> NDArray[np.float32]:
        """Calculate extraterrestrial radiation
        
        Args:
            dr: Relative earth-sun distance
            delta: Solar declination
            omega_s: Sunset hour angle
            
        Returns:
            Array of extraterrestrial radiation values
        """
        # Solar constant
        gsc = 0.0820  # MJ/m^2/min
        
        # Convert latitude to radians for calculation
        lat_rad = np.deg2rad(self.latitude)
        
        # Calculate Ra (MJ/m^2/day)
        ra = (24.0 * 60.0 / np.pi * gsc * dr * 
              (omega_s * np.sin(lat_rad) * np.sin(delta) +
               np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)))
        
        return ra
        
    def calculate_hargreaves(self, day_of_year: int, tmin: NDArray[np.float32],
                           tmax: NDArray[np.float32]) -> None:
        """Calculate reference ET using Hargreaves-Samani method
        
        Args:
            day_of_year: Day of year (1-365/366)
            tmin: Minimum temperature array (°F)
            tmax: Maximum temperature array (°F)
        """
        # Calculate mean temperature
        tmean = (tmax + tmin) / 2.0
        
        # Calculate temperature range
        tdelta = np.abs(tmax - tmin)
        
        # Get solar parameters
        dr, delta, omega_s = self._calculate_solar_parameters(day_of_year)
        
        # Calculate extraterrestrial radiation
        ra = self._calculate_extraterrestrial_radiation(dr, delta, omega_s)
        
        # Convert temperatures to Celsius
        tmean_c = (tmean - 32.0) * 5.0/9.0
        
        # Calculate reference ET (inches/day)
        self.reference_et0 = (self.hargreaves_params.et_slope * ra * 
                            (tmean_c + self.hargreaves_params.et_constant) * 
                            (tdelta ** self.hargreaves_params.et_exponent) / 25.4)  # Convert mm to inches
        
        # Ensure non-negative values
        self.reference_et0 = np.maximum(self.reference_et0, 0.0)
        
    def calculate_jensen_haise(self, day_of_year: int, tmin: NDArray[np.float32],
                             tmax: NDArray[np.float32], sun_pct: Optional[NDArray[np.float32]] = None) -> None:
        """Calculate reference ET using Jensen-Haise method
        
        Args:
            day_of_year: Day of year (1-365/366)
            tmin: Minimum temperature array (°F)
            tmax: Maximum temperature array (°F)
            sun_pct: Optional percent of possible sunshine
        """
        # Calculate mean temperature
        tmean = (tmax + tmin) / 2.0
        
        # Estimate sunshine percentage if not provided
        if sun_pct is None:
            sun_pct = 1.0 - ((tmax - tmin) / 20.0)  # Simple estimation
            sun_pct = np.clip(sun_pct, 0.3, 1.0)
        
        # Get solar parameters
        dr, delta, omega_s = self._calculate_solar_parameters(day_of_year)
        
        # Calculate extraterrestrial radiation
        ra = self._calculate_extraterrestrial_radiation(dr, delta, omega_s)
        
        # Calculate solar radiation
        rs = ra * (self.jensen_haise_params.as_coef + 
                  self.jensen_haise_params.bs_coef * sun_pct)
        
        # Convert mean temperature to Celsius
        tmean_c = (tmean - 32.0) * 5.0/9.0
        
        # Calculate reference ET (inches/day)
        self.reference_et0 = ((self.jensen_haise_params.ct_coef * tmean_c + 
                             self.jensen_haise_params.tx_coef) * rs / 25.4)  # Convert mm to inches
        
        # Set to zero when temperature is below freezing
        self.reference_et0[tmean <= 32.0] = 0.0
        
    def update_gridded(self, date: datetime, base_et: NDArray[np.float32]) -> None:
        """Update reference ET using gridded values/ratios
        
        Args:
            date: Current simulation date
            base_et: Base ET values to modify by ratios
        """
        if self.gridded_params is None:
            raise RuntimeError("Gridded parameters must be initialized first")
            
        # Update ratios on first day of month
        if date.day == 1:
            month_idx = date.month - 1  # Convert to 0-based index
            
            for zone_id, monthly_ratios in self.gridded_params.monthly_ratios.items():
                mask = self.gridded_params.et_zones == zone_id
                self.gridded_params.et_ratios[mask] = monthly_ratios[month_idx]
                
        # Apply ratios to base ET
        self.reference_et0 = base_et * self.gridded_params.et_ratios
        
    def calculate(self, date: datetime, tmin: NDArray[np.float32],
                 tmax: NDArray[np.float32], base_et: Optional[NDArray[np.float32]] = None,
                 sun_pct: Optional[NDArray[np.float32]] = None) -> NDArray[np.float32]:
        """Calculate reference ET using selected method
        
        Args:
            date: Current simulation date
            tmin: Minimum temperature array
            tmax: Maximum temperature array
            base_et: Optional base ET for gridded method
            sun_pct: Optional percent possible sunshine for Jensen-Haise
            
        Returns:
            Array of reference ET values
        """
        day_of_year = date.timetuple().tm_yday
        
        if self.method == "hargreaves":
            self.calculate_hargreaves(day_of_year, tmin, tmax)
        elif self.method == "jensen_haise":
            self.calculate_jensen_haise(day_of_year, tmin, tmax, sun_pct)
        elif self.method == "gridded":
            if base_et is None:
                raise ValueError("base_et must be provided when using gridded method")
            self.update_gridded(date, base_et)
            
        return self.reference_et0
