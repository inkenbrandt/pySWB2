from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
import numpy as np
from numpy.typing import NDArray
from pathlib import Path


from .precipitation import Precipitation, PrecipitationData
from .actual_et import ActualETCalculator
from .potential_et import PotentialETCalculator

@dataclass
class WeatherGridParams:
    """Parameters for gridded weather data"""
    zones: NDArray[np.int32]
    monthly_ratios: Dict[int, List[float]]  # Zone ID -> monthly adjustment ratios
    lapse_rate: float = 3.5  # Temperature lapse rate (°F per 1000 ft)
    precip_gradient: float = 0.0  # Precipitation gradient (in/1000 ft)

@dataclass
class FogParams:
    """Parameters for fog calculations"""
    min_temp_diff: float = 5.0  # Minimum temperature difference for fog (°F)
    max_deposition: float = 0.02  # Maximum fog deposition (inches)
    elevation_threshold: float = 1000.0  # Elevation threshold for fog (ft)

@dataclass
class WeatherAdjustments:
    """Parameters for weather data adjustments"""
    temperature_bias: float = 0.0  # Temperature adjustment (°F)
    precipitation_factor: float = 1.0  # Precipitation multiplier
    minimum_precip: float = 0.01  # Minimum precipitation threshold (inches)

class WeatherModule:
    """Module for handling all weather-related calculations"""
    
    def __init__(self, domain_size: int, grid_shape: tuple,
                 et_method: Literal["hargreaves", "jensen_haise", "gridded"] = "hargreaves",
                 actual_et_method: Literal["fao56", "fao56_two_stage", "thornthwaite"] = "fao56_two_stage"):
        """Initialize weather module
        
        Args:
            domain_size: Number of cells in model domain
            grid_shape: Shape of the model grid (rows, cols)
            et_method: Method for potential ET calculation
            actual_et_method: Method for actual ET calculation
        """
        self.domain_size = domain_size
        self.grid_shape = grid_shape
        
        # Initialize weather arrays
        self.tmin = np.zeros(domain_size, dtype=np.float32)
        self.tmax = np.zeros(domain_size, dtype=np.float32)
        self.tmean = np.zeros(domain_size, dtype=np.float32)
        self.elevation = np.zeros(domain_size, dtype=np.float32)
        
        # Initialize precipitation module
        self.precipitation = Precipitation(grid_shape)
        
        # Initialize fog array
        self.fog = np.zeros(domain_size, dtype=np.float32)
        
        # Initialize ET calculators
        self.potential_et = PotentialETCalculator(domain_size, et_method)
        self.actual_et = ActualETCalculator(domain_size, actual_et_method)
        
        # Parameters
        self.grid_params: Optional[WeatherGridParams] = None
        self.fog_params = FogParams()
        self.adjustments = WeatherAdjustments()
        
        # Current state tracking
        self.current_precip_data: Optional[PrecipitationData] = None
        
    def initialize(self, fragments_file: Path, rainfall_zones: NDArray[np.int32],
                  zones: NDArray[np.int32], monthly_ratios: Dict[int, List[float]],
                  elevation: NDArray[np.float32], latitude: NDArray[np.float32],
                  random_start: int = 12345) -> None:
        """Initialize weather module with all required data
        
        Args:
            fragments_file: Path to precipitation fragments file
            rainfall_zones: Array mapping cells to rainfall zones
            zones: Array mapping cells to weather zones
            monthly_ratios: Dictionary mapping zone IDs to monthly adjustment ratios
            elevation: Elevation array (feet)
            latitude: Latitude array (degrees)
            random_start: Random seed for precipitation fragments
        """
        # Initialize precipitation module
        self.precipitation.initialize(fragments_file, rainfall_zones, random_start)
        
        # Initialize grid parameters
        self.grid_params = WeatherGridParams(zones=zones, monthly_ratios=monthly_ratios)
        self.elevation = elevation
        self.potential_et.latitude = latitude
        
    def adjust_temperatures(self, base_tmin: NDArray[np.float32], 
                          base_tmax: NDArray[np.float32],
                          base_elevation: float) -> None:
        """Adjust temperatures for elevation and bias
        
        Args:
            base_tmin: Base minimum temperature array
            base_tmax: Base maximum temperature array
            base_elevation: Reference elevation for base temperatures
        """
        # Apply elevation lapse rate
        elevation_diff = (self.elevation - base_elevation) / 1000.0
        temp_adjustment = elevation_diff * self.grid_params.lapse_rate
        
        # Apply adjustments
        self.tmin = base_tmin - temp_adjustment + self.adjustments.temperature_bias
        self.tmax = base_tmax - temp_adjustment + self.adjustments.temperature_bias
        self.tmean = (self.tmin + self.tmax) / 2.0

    def process_precipitation(self, date: datetime, interception: NDArray[np.float32]) -> None:
        """Process precipitation calculations for current timestep

        Args:
            date: Current simulation date
            interception: Canopy interception array
        """
        # Calculate precipitation components using precipitation module
        self.current_precip_data = self.precipitation.calculate_precipitation(
            date=date,
            tmin=self.tmin,
            tmax=self.tmax,
            interception=interception
        )

        # Apply elevation adjustments if configured
        if hasattr(self, 'grid_params') and abs(self.grid_params.precip_gradient) > 1e-6:
            base_elevation = np.mean(self.elevation)
            elevation_diff = (self.elevation - base_elevation) / 1000.0
            precip_adjustment = elevation_diff * self.grid_params.precip_gradient

            # Adjust gross precipitation
            self.current_precip_data.gross_precip += precip_adjustment

            # Adjust rainfall/snowfall based on temperature
            snow_mask = self.tmax <= 32.0
            self.current_precip_data.snowfall[snow_mask] += precip_adjustment[snow_mask]
            self.current_precip_data.rainfall[~snow_mask] += precip_adjustment[~snow_mask]

            # Recalculate net values
            self.current_precip_data.net_snowfall = np.maximum(
                0.0,
                self.current_precip_data.snowfall - interception
            )
            self.current_precip_data.net_rainfall = np.maximum(
                0.0,
                self.current_precip_data.rainfall - interception
            )

        # Apply minimum threshold using array operations
        precip_mask = self.current_precip_data.gross_precip < self.adjustments.minimum_precip
        if np.any(precip_mask):
            # Zero out precipitation only where it's below threshold
            self.current_precip_data.gross_precip[precip_mask] = 0.0
            self.current_precip_data.rainfall[precip_mask] = 0.0
            self.current_precip_data.snowfall[precip_mask] = 0.0
            self.current_precip_data.net_rainfall[precip_mask] = 0.0
            self.current_precip_data.net_snowfall[precip_mask] = 0.0
            
    def calculate_snowmelt(self) -> NDArray[np.float32]:
        """Calculate potential snowmelt based on temperature"""
        return self.precipitation.calculate_snowmelt(self.tmin, self.tmax)
        
    def calculate_fog(self) -> None:
        """Calculate fog deposition based on temperature and elevation"""
        # Calculate temperature difference from dewpoint
        # Using simple approximation: dewpoint ≈ tmin
        temp_dewpoint_diff = self.tmean - self.tmin
        
        # Calculate potential fog based on temperature difference
        fog_potential = np.where(
            (temp_dewpoint_diff <= self.fog_params.min_temp_diff) &
            (self.elevation >= self.fog_params.elevation_threshold),
            self.fog_params.max_deposition * 
            (1.0 - temp_dewpoint_diff / self.fog_params.min_temp_diff),
            0.0
        )
        
        # Scale fog by elevation above threshold
        elevation_factor = np.clip(
            (self.elevation - self.fog_params.elevation_threshold) / 1000.0,
            0.0, 1.0
        )
        
        self.fog = fog_potential * elevation_factor
        
    def calculate_reference_et(self, date: datetime,
                             sun_pct: Optional[NDArray[np.float32]] = None,
                             base_et: Optional[NDArray[np.float32]] = None) -> None:
        """Calculate reference ET using configured method
        
        Args:
            date: Current simulation date
            sun_pct: Optional percent possible sunshine (for Jensen-Haise)
            base_et: Optional base ET values (for gridded method)
        """
        self.reference_et0 = self.potential_et.calculate(
            date=date,
            tmin=self.tmin,
            tmax=self.tmax,
            base_et=base_et,
            sun_pct=sun_pct
        )
        
    def calculate_actual_et(self, soil_storage: NDArray[np.float64],
                          soil_storage_max: NDArray[np.float32],
                          infiltration: NDArray[np.float32],
                          crop_etc: NDArray[np.float32],
                          **kwargs) -> NDArray[np.float64]:
        """Calculate actual ET using configured method
        
        Args:
            soil_storage: Current soil moisture storage
            soil_storage_max: Maximum soil moisture storage
            infiltration: Infiltration amount
            crop_etc: Crop ET
            **kwargs: Additional arguments for specific ET methods
            
        Returns:
            Array of actual ET values
        """
        return self.actual_et.calculate(
            soil_storage=soil_storage,
            soil_storage_max=soil_storage_max,
            infiltration=infiltration,
            crop_etc=crop_etc,
            reference_et0=self.reference_et0,
            **kwargs
        )
        
    def process_timestep(self, date: datetime,
                        base_tmin: NDArray[np.float32],
                        base_tmax: NDArray[np.float32],
                        base_elevation: float,
                        interception: NDArray[np.float32],
                        **kwargs) -> None:
        """Process all weather calculations for current timestep
        
        Args:
            date: Current simulation date
            base_tmin: Base minimum temperature array
            base_tmax: Base maximum temperature array
            base_elevation: Reference elevation for base data
            interception: Canopy interception array
            **kwargs: Additional arguments for ET calculations
        """
        # Adjust temperatures
        self.adjust_temperatures(base_tmin, base_tmax, base_elevation)
        
        # Process precipitation
        self.process_precipitation(date, interception)
        
        # Calculate other components
        self.calculate_fog()
        self.calculate_reference_et(date)
        
        if kwargs.get('calculate_actual_et', False):
            required_args = ['soil_storage', 'soil_storage_max', 'infiltration', 'crop_etc']
            if not all(arg in kwargs for arg in required_args):
                raise ValueError(f"Missing required arguments for actual ET calculation: "
                              f"{[arg for arg in required_args if arg not in kwargs]}")
            
            self.calculate_actual_et(
                soil_storage=kwargs['soil_storage'],
                soil_storage_max=kwargs['soil_storage_max'],
                infiltration=kwargs['infiltration'],
                crop_etc=kwargs['crop_etc'],
                **{k: v for k, v in kwargs.items() if k not in required_args}
            )
