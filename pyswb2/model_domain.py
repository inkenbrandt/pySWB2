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

        # Routing configuration
        self.routing_enabled: bool = False
        self.flow_direction: Optional[NDArray[np.int32]] = None
        self.flow_sequence: Optional[NDArray[np.int32]] = None
        self.routing_fraction: Optional[NDArray[np.float32]] = None

        # Time tracking
        self.current_date: Optional[datetime] = None
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None

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

        # Crop growth settings
        self.dynamic_rooting: bool = False
        self.use_crop_coefficients: bool = False

        # Output specifications
        self.outspecs: Dict[OutputType, OutputSpec] = {}
        self._initialize_output_specs()

    def initialize_simulation_period(self, start_date: datetime, end_date: datetime) -> None:
        """Initialize simulation start and end dates

        Args:
            start_date: Simulation start date
            end_date: Simulation end date
        """
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date

    def update_date(self, date: datetime) -> None:
        """Update current simulation date

        Args:
            date: Current simulation date
        """
        self.current_date = date

        # Update date in all modules that need it
        if hasattr(self, 'agriculture_module') and self.agriculture_module is not None:
            self.agriculture_module.current_date = date

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
        # storage components
        self.surface_storage = None
        self.snow_storage = None
        self.interception_storage = None
        self.interception_storage_max = None  # Add max interception storage
        self.interception = None  # Add interception array
        self.soil_storage = None
        self.soil_storage_max = None

        # precip arrays
        self.gross_precip = None
        self.rainfall = None
        self.snowfall = None
        self.snowmelt = None
        self.runoff = None
        self.runon = None
        self.infiltration = None
        self.net_infiltration = None
        self.reference_et0 = None
        self.actual_et = None

        # Energy and ET arrays
        self.reference_et0 = None
        self.actual_et = None
        self.tmin = None
        self.tmax = None
        self.tmean = None
        self.fog = None

        # Crop-related arrays
        self.crop_coefficient = None  # Current crop coefficient value
        self.crop_etc = None  # Crop ET under standard conditions
        self.rooting_depth = None  # Current rooting depth
        self.max_root_depth = None  # Maximum rooting depth by land use

        # Soil properties
        self.field_capacity = None  # Field capacity by soil type
        self.wilting_point = None  # Wilting point by soil type
        self.direct_soil_moisture = None  # Add direct soil moisture array
        self.direct_net_infiltration = None  # Added direct net infiltration array
        self.direct_soil_moisture = None     # Added direct soil moisture array for completeness

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
        self.interception_storage_max = np.zeros(n_active, dtype=np.float32)

        # Water fluxes
        self.gross_precip = np.zeros(n_active, dtype=np.float32)
        self.rainfall = np.zeros(n_active, dtype=np.float32)
        self.snowfall = np.zeros(n_active, dtype=np.float32)
        self.snowmelt = np.zeros(n_active, dtype=np.float32)
        self.runoff = np.zeros(n_active, dtype=np.float32)
        self.runon = np.zeros(n_active, dtype=np.float32)
        self.infiltration = np.zeros(n_active, dtype=np.float32)
        self.net_infiltration = np.zeros(n_active, dtype=np.float32)
        self.fog = np.zeros(n_active, dtype=np.float32)
        self.interception = np.zeros(n_active, dtype=np.float32)  # Initialize interception array
        self.direct_soil_moisture = np.zeros(n_active, dtype=np.float32)  # Initialize direct soil moisture

        # Energy and ET
        self.reference_et0 = np.zeros(n_active, dtype=np.float32)
        self.actual_et = np.zeros(n_active, dtype=np.float32)
        self.tmin = np.zeros(n_active, dtype=np.float32)
        self.tmax = np.zeros(n_active, dtype=np.float32)
        self.tmean = np.zeros(n_active, dtype=np.float32)

        # Crop and soil parameters
        self.crop_coefficient = np.ones(n_active, dtype=np.float32)  # Initialize to 1.0
        self.crop_etc = np.zeros(n_active, dtype=np.float32)
        self.rooting_depth = np.zeros(n_active, dtype=np.float32)
        self.max_root_depth = np.zeros(n_active, dtype=np.float32)
        self.field_capacity = np.zeros(n_active, dtype=np.float32)
        self.wilting_point = np.zeros(n_active, dtype=np.float32)

        # Direct additions
        self.direct_net_infiltration = np.zeros(n_active, dtype=np.float32)
        self.direct_soil_moisture = np.zeros(n_active, dtype=np.float32)

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
        self.current_date = date

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

    def calc_reference_et(self) -> None:
        """Calculate reference evapotranspiration"""
        if self.current_date is None:
            raise RuntimeError("Current date must be set before calculating reference ET")

        self.reference_et0 = self.potential_et_calculator.calculate(
            date=self.current_date,
            tmin=self.tmin,
            tmax=self.tmax
        )

    def initialize_crop_coefficients(self, use_crop_coefficients: bool = False,
                                     dynamic_rooting: bool = False) -> None:
        """Initialize crop coefficient settings

        Args:
            use_crop_coefficients: Whether to use FAO-56 crop coefficients
            dynamic_rooting: Whether to use dynamic rooting depths
        """
        self.use_crop_coefficients = use_crop_coefficients
        self.dynamic_rooting = dynamic_rooting

        # Initialize crop coefficient to 1.0 if not using FAO-56
        if not use_crop_coefficients:
            self.crop_coefficient = np.ones_like(self.crop_coefficient)

    def initialize_routing(self, flow_direction: NDArray[np.int32],
                           routing_fraction: Optional[NDArray[np.float32]] = None) -> None:
        """Initialize routing configuration

        Args:
            flow_direction: Array defining D8 flow direction for each cell
            routing_fraction: Optional array defining fraction of runoff to route (default 1.0)
        """
        if flow_direction.shape != (self.domain_size,):
            raise ValueError("Flow direction array must match domain size")

        self.flow_direction = flow_direction

        # Default routing fraction to 1.0 if not provided
        if routing_fraction is None:
            self.routing_fraction = np.ones(self.domain_size, dtype=np.float32)
        else:
            if routing_fraction.shape != (self.domain_size,):
                raise ValueError("Routing fraction array must match domain size")
            self.routing_fraction = routing_fraction

        # Create sequence of cells from upslope to downslope
        self._create_flow_sequence()

        # Enable routing
        self.routing_enabled = True

    def _create_flow_sequence(self) -> None:
        """Create sequence of cell indices from upslope to downslope

        This creates an ordered list of cell indices such that each cell
        appears after all cells that could potentially flow into it.
        """
        # Initialize visited flags and sequence
        n_cells = len(self.flow_direction)
        visited = np.zeros(n_cells, dtype=bool)
        sequence = []

        # Process each unvisited cell
        for i in range(n_cells):
            if not visited[i]:
                self._trace_flow_path(i, visited, sequence)

        # Store final sequence
        self.flow_sequence = np.array(sequence, dtype=np.int32)

    def _trace_flow_path(self, cell: int, visited: NDArray[np.bool_],
                         sequence: List[int]) -> None:
        """Recursively trace flow path from a cell

        Args:
            cell: Current cell index
            visited: Array tracking visited cells
            sequence: List to store cell sequence
        """
        # Mark current cell as visited
        visited[cell] = True

        # Get next downslope cell
        next_cell = self.flow_direction[cell]

        # Follow flow path if valid cell and not already visited
        if next_cell >= 0 and not visited[next_cell]:
            self._trace_flow_path(next_cell, visited, sequence)

        # Add current cell to sequence
        sequence.append(cell)

    def set_crop_options(self, dynamic_rooting: bool = False,
                         use_crop_coefficients: bool = False) -> None:
        """Set crop growth calculation options

        Args:
            dynamic_rooting: Whether to use dynamic root depth calculations
            use_crop_coefficients: Whether to use FAO-56 crop coefficients
        """
        self.dynamic_rooting = dynamic_rooting
        self.use_crop_coefficients = use_crop_coefficients