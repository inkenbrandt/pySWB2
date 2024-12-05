from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import netCDF4 as nc
import os

from .runoff import RunoffModule, RunoffParameters
from .agriculture import AgricultureModule, CropParameters
from .actual_et import ActualETCalculator
from .interception import InterceptionModule, GashParameters, BucketParameters
from .infiltration import InfiltrationModule, InfiltrationParameters
from .soil import SoilModule, SoilParameters
from .weather import WeatherModule
from .potential_et import PotentialETCalculator

class OutputType(Enum):
    GROSS_PRECIPITATION = 1
    RAINFALL = 2
    SNOWFALL = 3
    INTERCEPTION = 4
    INTERCEPTION_STORAGE = 5
    RUNON = 6
    RUNOFF = 7
    SNOW_STORAGE = 8
    SURFACE_STORAGE = 9
    SOIL_STORAGE = 10
    DELTA_SOIL_STORAGE = 11
    REFERENCE_ET0 = 12
    ACTUAL_ET = 13
    SNOWMELT = 14
    TMIN = 15
    TMAX = 16
    NET_INFILTRATION = 17
    REJECTED_NET_INFILTRATION = 18
    INFILTRATION = 19
    IRRIGATION = 20
    RUNOFF_OUTSIDE = 21
    CROP_ET = 22
    BARE_SOIL_EVAP = 23
    GROWING_DEGREE_DAY = 24
    DIRECT_NET_INFILTRATION = 25
    DIRECT_SOIL_MOISTURE = 26
    STORM_DRAIN_CAPTURE = 27
    GROWING_SEASON = 28
    FOG = 29

@dataclass
class OutputSpec:
    variable_name: str
    variable_units: str
    valid_minimum: float
    valid_maximum: float
    is_active: bool = False
    multisim_outputs: bool = False

class ModelDomain:
    """Class representing model domain and calculations"""
    
    def __init__(self):
        # Grid properties
        self.number_of_columns: int = 0
        self.number_of_rows: int = 0
        self.gridcellsize: float = 0.0
        self.x_ll: float = 0.0
        self.y_ll: float = 0.0
        self.proj4_string: str = ""
        
        # Initialize component modules
        self.domain_size = 0  # Will be set in initialize_grid
        self._initialize_modules()
        
        # Masks and indices
        self.active: NDArray[np.bool_] = None
        self.landuse_code: NDArray[np.int32] = None
        self.landuse_index: NDArray[np.int32] = None
        self.soil_group: NDArray[np.int32] = None
        
        # State variables
        self._initialize_state_arrays()
        
        # Output specifications
        self.outspecs: Dict[OutputType, OutputSpec] = {}
        self._initialize_output_specs()

    def _initialize_output_specs(self):
        """Initialize output specifications with default values"""
        self.outspecs[OutputType.GROSS_PRECIPITATION] = OutputSpec(
            variable_name="gross_precipitation",
            variable_units="inches",
            valid_minimum=0.0,
            valid_maximum=100.0
        )
        self.outspecs[OutputType.NET_INFILTRATION] = OutputSpec(
            variable_name="net_infiltration",
            variable_units="inches",
            valid_minimum=0.0,
            valid_maximum=100.0
        )
        self.outspecs[OutputType.SOIL_STORAGE] = OutputSpec(
            variable_name="soil_storage",
            variable_units="inches",
            valid_minimum=0.0,
            valid_maximum=100.0
        )
        self.outspecs[OutputType.ACTUAL_ET] = OutputSpec(
            variable_name="actual_et",
            variable_units="inches",
            valid_minimum=0.0,
            valid_maximum=20.0
        )
        self.outspecs[OutputType.SNOWMELT] = OutputSpec(
            variable_name="snowmelt",
            variable_units="inches",
            valid_minimum=0.0,
            valid_maximum=100.0
        )
        
    def _initialize_modules(self):
        """Initialize all component modules"""
        # These will be properly initialized once domain_size is known
        self.runoff_module: Optional[RunoffModule] = None
        self.agriculture_module: Optional[AgricultureModule] = None
        self.actual_et_calculator: Optional[ActualETCalculator] = None
        self.interception_module: Optional[InterceptionModule] = None
        self.infiltration_module: Optional[InfiltrationModule] = None
        self.soil_module: Optional[SoilModule] = None
        self.weather_module: Optional[WeatherModule] = None
        self.potential_et_calculator: Optional[PotentialETCalculator] = None

    def _initialize_state_arrays(self):
        """Initialize state tracking arrays"""
        # These will be resized once domain_size is known
        self.soil_storage = None
        self.soil_storage_max = None
        self.surface_storage = None
        self.snow_storage = None
        self.interception_storage = None
        self.gross_precip = None
        self.rainfall = None
        self.snowfall = None
        self.runoff = None
        self.runon = None
        self.infiltration = None
        self.net_infiltration = None
        self.reference_et0 = None
        self.actual_et = None
        self.tmin = None
        self.tmax = None
        self.tmean = None
        self.fog = None  # Add fog array

    def initialize_grid(self, nx: int, ny: int, x_ll: float, y_ll: float, 
                       cell_size: float, proj4: str = ""):
        """Initialize model grid properties"""
        self.number_of_columns = nx
        self.number_of_rows = ny
        self.gridcellsize = cell_size
        self.x_ll = x_ll
        self.y_ll = y_ll
        self.proj4_string = proj4
        
        # Initialize domain size and masks
        self.domain_size = nx * ny
        self.active = np.ones((ny, nx), dtype=bool)
        
        # Initialize modules with domain size
        self.runoff_module = RunoffModule(self.domain_size)
        self.agriculture_module = AgricultureModule(self.domain_size)
        self.actual_et_calculator = ActualETCalculator(self.domain_size)
        self.interception_module = InterceptionModule(self.domain_size)
        self.infiltration_module = InfiltrationModule(self.domain_size)
        self.soil_module = SoilModule(self.domain_size)
        self.weather_module = WeatherModule(self.domain_size, (ny, nx))
        self.potential_et_calculator = PotentialETCalculator(self.domain_size)
        
        # Initialize state arrays with proper size
        self._initialize_domain_arrays()

    def _initialize_domain_arrays(self):
        """Initialize all domain arrays to proper size"""
        n_active = np.count_nonzero(self.active)
        
        # Initialize arrays
        self.landuse_code = np.zeros(n_active, dtype=np.int32)
        self.landuse_index = np.zeros(n_active, dtype=np.int32)
        self.soil_group = np.zeros(n_active, dtype=np.int32)
        
        # Water storages
        self.soil_storage = np.zeros(n_active, dtype=np.float32)
        self.soil_storage_max = np.zeros(n_active, dtype=np.float32)
        self.surface_storage = np.zeros(n_active, dtype=np.float32)
        self.snow_storage = np.zeros(n_active, dtype=np.float32)
        self.interception_storage = np.zeros(n_active, dtype=np.float32)
        
        # Water fluxes
        self.gross_precip = np.zeros(n_active, dtype=np.float32)
        self.rainfall = np.zeros(n_active, dtype=np.float32)
        self.snowfall = np.zeros(n_active, dtype=np.float32)
        self.runoff = np.zeros(n_active, dtype=np.float32)
        self.runon = np.zeros(n_active, dtype=np.float32)
        self.infiltration = np.zeros(n_active, dtype=np.float32)
        self.net_infiltration = np.zeros(n_active, dtype=np.float32)
        self.fog = np.zeros(n_active, dtype=np.float32)  # Initialize fog array
        
        # Energy and ET
        self.reference_et0 = np.zeros(n_active, dtype=np.float32)
        self.actual_et = np.zeros(n_active, dtype=np.float32)
        self.tmin = np.zeros(n_active, dtype=np.float32)
        self.tmax = np.zeros(n_active, dtype=np.float32)
        self.tmean = np.zeros(n_active, dtype=np.float32)

    def initialize_parameters(self, landuse_params: Dict[int, Dict], 
                            soil_params: Dict[int, Dict]) -> None:
        """Initialize parameters for all modules
        
        Args:
            landuse_params: Dictionary mapping landuse IDs to parameter dictionaries
            soil_params: Dictionary mapping soil IDs to parameter dictionaries
        """
        for landuse_id, params in landuse_params.items():
            # Add parameters to each module
            self.runoff_module.add_parameters(landuse_id, 
                RunoffParameters(**params.get('runoff', {})))
            
            self.agriculture_module.add_crop_parameters(landuse_id,
                CropParameters(**params.get('crop', {})))
            
            self.interception_module.add_gash_parameters(landuse_id,
                GashParameters(**params.get('interception', {})))
                
        for soil_id, params in soil_params.items():
            self.infiltration_module.add_soil_parameters(soil_id,
                InfiltrationParameters(**params.get('infiltration', {})))
            
            self.soil_module.add_soil_parameters(soil_id,
                SoilParameters(**params.get('soil', {})))

    def initialize_modules(self, landuse_indices: NDArray[np.int32],
                         soil_indices: NDArray[np.int32],
                         elevation: NDArray[np.float32],
                         latitude: NDArray[np.float32],
                         fragments_file: Optional[str] = None,
                           ) -> None:
        """Initialize all modules with required data
        
        Args:
            landuse_indices: Array mapping cells to landuse types
            soil_indices: Array mapping cells to soil types
            elevation: Elevation array
            latitude: Latitude array
        """
        # Initialize each module
        self.runoff_module.initialize(landuse_indices)
        self.agriculture_module.initialize(landuse_indices)
        self.soil_module.initialize(soil_indices, self.soil_storage_max)
        self.infiltration_module.initialize(soil_indices, self.soil_storage_max)
        
        # Set up weather module with required data
        self.weather_module.initialize(
            fragments_file=fragments_file,  # This would need to be provided
            rainfall_zones=np.ones_like(landuse_indices),  # Default to single zone
            zones=np.ones_like(landuse_indices),
            monthly_ratios={1: [1.0] * 12},  # Default to no monthly adjustments
            elevation=elevation,
            latitude=latitude
        )

    def get_weather_data(self, date: datetime) -> None:
        """Update weather data for given date"""
        self.weather_module.process_timestep(
            date=date,
            base_tmin=self.tmin,
            base_tmax=self.tmax,
            base_elevation=0.0,  # This would need to be provided
            interception=self.interception_storage
        )
        
        # Update local arrays
        self.tmin = self.weather_module.tmin
        self.tmax = self.weather_module.tmax
        self.tmean = self.weather_module.tmean
        self.reference_et0 = self.weather_module.reference_et0
        
        if self.weather_module.current_precip_data:
            self.gross_precip = self.weather_module.current_precip_data.gross_precip
            self.rainfall = self.weather_module.current_precip_data.rainfall
            self.snowfall = self.weather_module.current_precip_data.snowfall

    def process_daily_timestep(self, date: datetime) -> None:
        """Process all calculations for current timestep
        
        Args:
            date: Current simulation date
        """
        # Update agricultural conditions
        self.agriculture_module.process_daily(
            date, self.tmean, self.tmin, self.tmax
        )
        
        # Calculate interception
        self.interception_module.calculate_interception(
            self.rainfall, 
            self.weather_module.fog,
            self.agriculture_module.is_growing_season
        )
        
        # Calculate infiltration
        self.infiltration_module.process_timestep(
            self.soil_storage,
            self.gross_precip,
            self.runoff,
            np.zeros_like(self.gross_precip)  # CFGI change would be calculated
        )
        
        # Calculate runoff
        self.runoff_module.process_timestep(
            date,
            self.gross_precip,
            self.agriculture_module.is_growing_season
        )
        
        # Update soil moisture
        self.soil_module.process_timestep(
            date,
            self.gross_precip,
            self.actual_et,
            self.reference_et0
        )
        
        # Calculate actual ET
        self.actual_et = self.actual_et_calculator.calculate(
            self.soil_storage,
            self.soil_storage_max,
            self.infiltration,
            self.reference_et0
        )
