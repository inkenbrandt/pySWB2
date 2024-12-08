from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import logging

from .runoff import RunoffModule, RunoffParameters
from .agriculture import AgricultureModule, CropParameters
from .actual_et import ActualETCalculator, FAO56Parameters
from .interception import InterceptionModule, GashParameters
from .infiltration import InfiltrationModule, InfiltrationParameters
from .soil import SoilModule, SoilParameters
from .weather import WeatherModule
from .potential_et import PotentialETCalculator
from .logging_config import ParameterLogger


class ModelDomain:
    """Core model domain class handling state, calculations, and module coordination"""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize model domain with logging"""
        # Initialize logger
        self.logger = ParameterLogger(log_dir)

        # State tracking
        self.is_initialized: bool = False
        self.parameters_initialized: bool = False
        self.output_initialized: bool = False

        # Grid properties
        self.number_of_columns: int = 0
        self.number_of_rows: int = 0
        self.gridcellsize: float = 0.0
        self.x_ll: float = 0.0
        self.y_ll: float = 0.0
        self.proj4_string: str = ""
        self.domain_size: int = 0
        self.active: Optional[NDArray[np.bool_]] = None

        # Time tracking
        self.current_date: Optional[datetime] = None
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None

        # Initialize empty arrays
        self._initialize_arrays()

        # Module instances (initialized later)
        self.runoff_module: Optional[RunoffModule] = None
        self.agriculture_module: Optional[AgricultureModule] = None
        self.actual_et_calculator: Optional[ActualETCalculator] = None
        self.interception_module: Optional[InterceptionModule] = None
        self.infiltration_module: Optional[InfiltrationModule] = None
        self.soil_module: Optional[SoilModule] = None
        self.weather_module: Optional[WeatherModule] = None
        self.potential_et_calculator: Optional[PotentialETCalculator] = None

        # Configuration flags
        self.routing_enabled: bool = False
        self.dynamic_rooting: bool = False
        self.use_crop_coefficients: bool = False

        # Routing variables
        self.flow_direction: Optional[NDArray[np.int32]] = None
        self.flow_sequence: Optional[NDArray[np.int32]] = None
        self.routing_fraction: Optional[NDArray[np.float32]] = None

        # Module indices
        self.landuse_indices: Optional[NDArray[np.int32]] = None
        self.soil_indices: Optional[NDArray[np.int32]] = None

    def _initialize_arrays(self) -> None:
        """Initialize empty arrays"""
        # Storage variables
        self.surface_storage = None
        self.snow_storage = None
        self.interception_storage = None
        self.soil_storage = None
        self.soil_storage_max = None

        # Precipitation and moisture variables
        self.gross_precip = None
        self.rainfall = None
        self.snowfall = None
        self.net_rainfall = None
        self.net_snowfall = None
        self.fog = None

        # Water flux variables
        self.runoff = None
        self.runon = None
        self.infiltration = None
        self.net_infiltration = None
        self.potential_snowmelt = None
        self.snowmelt = None
        self.direct_net_infiltration = None
        self.direct_soil_moisture = None

        # ET variables
        self.reference_et0 = None
        self.actual_et = None
        self.actual_et_interception = None
        self.crop_etc = None

        # Interception variables
        self.interception = None
        self.interception_storage = None
        self.interception_storage_max = None

        # Temperature variables
        self.tmin = None
        self.tmax = None
        self.tmean = None

        # Vegetation variables
        self.crop_coefficient = None
        self.rooting_depth = None
        self.wilting_point = None
        self.field_capacity = None
        self.available_water_content = None

    def _initialize_domain_arrays(self) -> None:
        """Initialize domain arrays to proper size"""
        n_active = np.count_nonzero(self.active)

        # Create all arrays with proper size
        array_names = [
            'surface_storage', 'snow_storage', 'interception_storage',
            'soil_storage', 'soil_storage_max', 'gross_precip',
            'rainfall', 'snowfall', 'net_rainfall', 'net_snowfall',
            'fog', 'runoff', 'runon', 'infiltration', 'net_infiltration',
            'potential_snowmelt', 'snowmelt', 'direct_net_infiltration',
            'direct_soil_moisture', 'reference_et0', 'actual_et',
            'actual_et_interception', 'crop_etc', 'interception',
            'interception_storage', 'interception_storage_max',
            'tmin', 'tmax', 'tmean', 'crop_coefficient', 'rooting_depth',
            'wilting_point', 'field_capacity', 'available_water_content'
        ]

        for name in array_names:
            if name == 'crop_coefficient':
                # Initialize crop coefficient to 1.0
                setattr(self, name, np.ones(n_active, dtype=np.float32))
            else:
                # Initialize other arrays to 0.0
                setattr(self, name, np.zeros(n_active, dtype=np.float32))

        self.logger.debug(f"Initialized {len(array_names)} domain arrays with size {n_active}")

    def initialize_grid(self, nx: int, ny: int, x_ll: float, y_ll: float,
                        cell_size: float, proj4: str = "") -> None:
        """Initialize model grid properties"""
        self.logger.info("Initializing model grid")

        # Validate inputs
        if nx <= 0 or ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if cell_size <= 0:
            raise ValueError("Cell size must be positive")

        # Set grid properties
        self.number_of_columns = nx
        self.number_of_rows = ny
        self.gridcellsize = cell_size
        self.x_ll = x_ll
        self.y_ll = y_ll
        self.proj4_string = proj4

        # Initialize domain size and active cells mask
        self.domain_size = nx * ny
        self.active = np.ones((ny, nx), dtype=bool)

        # Initialize arrays to proper size
        self._initialize_domain_arrays()

        self.is_initialized = True
        self.logger.info(f"Grid initialized: {nx}x{ny} cells")

    def initialize_parameters(self, landuse_params: Dict[int, Dict],
                              soil_params: Dict[int, Dict]) -> None:
        """Initialize parameters for all modules"""
        if not self.is_initialized:
            raise RuntimeError("Grid must be initialized before parameters")

        self.logger.info("Initializing model parameters")

        try:
            # Initialize module parameters
            self._initialize_modules()

            # Add a default entry to landuse_params if not present
            if 0 not in landuse_params:
                landuse_params[0] = {
                    'runoff': {
                        'curve_number': 70.0,
                        'initial_abstraction_ratio': 0.2,
                        'depression_storage': 0.1,
                        'impervious_fraction': 0.0
                    },
                    'crop': {
                        'gdd_base': 50.0,
                        'gdd_max': 86.0,
                        'growing_season_start': None,
                        'growing_season_end': None,
                        'initial_root_depth': 50.0,
                        'max_root_depth': 1500.0,
                        'crop_coefficient': 1.0
                    },
                    'interception': {
                        'canopy_storage_capacity': 0.1,
                        'trunk_storage_capacity': 0.0,
                        'stemflow_fraction': 0.0,
                        'interception_storage_max_growing': 0.1,
                        'interception_storage_max_nongrowing': 0.1
                    }
                }

            # Initialize landuse parameters
            for landuse_id, params in landuse_params.items():
                if 'runoff' in params:
                    self.runoff_module.add_parameters(
                        landuse_id, RunoffParameters(**params['runoff']))
                if 'crop' in params:
                    self.agriculture_module.add_crop_parameters(
                        landuse_id, CropParameters(**params['crop']))
                if 'interception' in params:
                    self.interception_module.add_gash_parameters(
                        landuse_id, GashParameters(**params['interception']))

                # Add ET parameters using crop parameters
                if 'crop' in params:
                    et_params = FAO56Parameters(
                        depletion_fraction=0.5,  # Default value
                        rew=0.1,  # Default REW
                        tew=0.4,  # Default TEW
                        mean_plant_height=params['crop'].get('max_root_depth', 0.0) / 12.0,  # Convert to feet
                        kcb_min=0.15,  # Default minimum
                        kcb_mid=params['crop'].get('crop_coefficient', 1.0),
                        kcb_end=0.15,  # Default end value
                        initial_root_depth=params['crop'].get('initial_root_depth', 50.0),
                        max_root_depth=params['crop'].get('max_root_depth', 1500.0)
                    )
                    self.actual_et_calculator.add_parameters(landuse_id, et_params)

            self.parameters_initialized = True
            self.logger.info("Parameter initialization complete")

        except Exception as e:
            self.logger.error(f"Parameter initialization failed: {str(e)}")
            raise

    def initialize_modules(self, landuse_indices: NDArray[np.int32],
                           soil_indices: NDArray[np.int32],
                           elevation: NDArray[np.float32],
                           latitude: NDArray[np.float32],
                           fragments_file: Optional[Path] = None) -> None:
        """Initialize all modules with required data"""
        if not self.parameters_initialized:
            raise RuntimeError("Parameters must be initialized before modules")

        self.logger.info("Initializing model modules")

        try:
            # Store indices
            self.landuse_indices = landuse_indices
            self.soil_indices = soil_indices

            # Initialize each module
            self.runoff_module.initialize(landuse_indices)
            self.agriculture_module.initialize(landuse_indices)
            self.soil_module.initialize(soil_indices, self.soil_storage_max)
            self.infiltration_module.initialize(soil_indices, self.soil_storage_max)

            # Initialize interception module with landuse data and default canopy cover
            self.interception_module.initialize(
                landuse_indices=landuse_indices,
                canopy_cover=np.ones_like(landuse_indices, dtype=np.float32),  # Default full cover
                evap_to_rain_ratio=np.ones_like(landuse_indices, dtype=np.float32)  # Default 1:1 ratio
            )

            # Initialize weather module
            self.weather_module.initialize(
                fragments_file=fragments_file,
                rainfall_zones=np.ones_like(landuse_indices),
                zones=np.ones_like(landuse_indices),
                monthly_ratios={1: [1.0] * 12},
                elevation=elevation,
                latitude=latitude
            )

            # Initialize potential ET calculator with latitude
            self.potential_et_calculator.latitude = latitude

            # Initialize actual ET calculator with relevant data
            self.actual_et_calculator.initialize(
                landuse_indices=landuse_indices,
                soil_indices=soil_indices,
                elevation=elevation
            )

            self.logger.info("Module initialization complete")

        except Exception as e:
            self.logger.error(f"Module initialization failed: {str(e)}")
            raise

    def _initialize_modules(self) -> None:
        """Initialize all model modules"""
        self.runoff_module = RunoffModule(self.domain_size)
        self.agriculture_module = AgricultureModule(self.domain_size)
        self.actual_et_calculator = ActualETCalculator(self.domain_size)
        self.interception_module = InterceptionModule(self.domain_size)
        self.infiltration_module = InfiltrationModule(self.domain_size)
        self.soil_module = SoilModule(self.domain_size)
        self.weather_module = WeatherModule(self.domain_size, (self.number_of_rows, self.number_of_columns))
        self.potential_et_calculator = PotentialETCalculator(self.domain_size)

    def initialize_simulation_period(self, start_date: datetime, end_date: datetime) -> None:
        """Initialize simulation start and end dates"""
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date

        self.logger.info(f"Simulation period: {start_date} to {end_date}")

    def initialize_output(self, directory: Path, prefix: str,
                          variables: List[str], compression: bool = True) -> None:
        """Initialize output configuration"""
        if not self.is_initialized:
            raise RuntimeError("Domain must be initialized before output")

        self.output_dir = directory
        self.output_prefix = prefix
        self.output_variables = variables
        self.output_compression = compression

        # Create output directory
        directory.mkdir(parents=True, exist_ok=True)

        self.output_initialized = True
        self.logger.info("Output configuration initialized")

    def update_date(self, date: datetime) -> None:
        """Update current simulation date"""
        self.current_date = date

        if self.agriculture_module is not None:
            self.agriculture_module.current_date = date

    def get_weather_data(self, date: datetime) -> None:
        """Update weather data for current date"""
        self.weather_module.process_timestep(
            date=date,
            base_tmin=self.tmin,
            base_tmax=self.tmax,
            base_elevation=0.0,
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

    def initialize_crop_coefficients(self, use_crop_coefficients: bool = False,
                                     dynamic_rooting: bool = False) -> None:
        """Initialize crop coefficient settings"""
        self.use_crop_coefficients = use_crop_coefficients
        self.dynamic_rooting = dynamic_rooting

        if not use_crop_coefficients:
            self.crop_coefficient = np.ones_like(self.crop_coefficient)

    def calc_reference_et(self) -> None:
        """Calculate reference evapotranspiration"""
        if self.current_date is None:
            raise RuntimeError("Current date must be set before calculating reference ET")

        self.reference_et0 = self.potential_et_calculator.calculate(
            date=self.current_date,
            tmin=self.tmin,
            tmax=self.tmax
        )

    def initialize_routing(self, flow_direction: NDArray[np.int32],
                           routing_fraction: Optional[NDArray[np.float32]] = None) -> None:
        """Initialize routing configuration"""
        if flow_direction.shape != (self.domain_size,):
            raise ValueError("Flow direction array must match domain size")

        self.flow_direction = flow_direction
        self.routing_fraction = np.ones(self.domain_size,
                                        dtype=np.float32) if routing_fraction is None else routing_fraction
        self._create_flow_sequence()
        self.routing_enabled = True

    def _create_flow_sequence(self) -> None:
        """Create sequence of cell indices from upslope to downslope"""
        n_cells = len(self.flow_direction)
        visited = np.zeros(n_cells, dtype=bool)
        sequence = []

        for i in range(n_cells):
            if not visited[i]:
                self._trace_flow_path(i, visited, sequence)

        self.flow_sequence = np.array(sequence, dtype=np.int32)

    def _trace_flow_path(self, cell: int, visited: NDArray[np.bool_],
                         sequence: List[int]) -> None:
        """Recursively trace flow path from a cell"""
        visited[cell] = True
        next_cell = self.flow_direction[cell]

        if next_cell >= 0 and not visited[next_cell]:
            self._trace_flow_path(next_cell, visited, sequence)

        sequence.append(cell)

    def cleanup(self) -> None:
        """Clean up model resources"""
        self.logger.info("Cleaning up model resources")

        try:
            # Clean up modules
            modules = [
                'runoff_module', 'agriculture_module', 'actual_et_calculator',
                'interception_module', 'infiltration_module', 'soil_module',
                'weather_module', 'potential_et_calculator'
            ]

            for module_name in modules:
                module = getattr(self, module_name, None)
                if module is not None and hasattr(module, 'cleanup'):
                    try:
                        module.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up {module_name}: {str(e)}")

            # Clear arrays
            self._clear_arrays()

            # Clean up logger
            handlers = self.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.logger.removeHandler(handler)

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

    def _clear_arrays(self) -> None:
        """Clear memory by setting arrays to None"""
        array_names = [
            'surface_storage', 'snow_storage', 'interception_storage',
            'soil_storage', 'soil_storage_max', 'gross_precip',
            'rainfall', 'snowfall', 'net_rainfall', 'net_snowfall',
            'fog', 'runoff', 'runon', 'infiltration', 'net_infiltration',
            'potential_snowmelt', 'snowmelt', 'direct_net_infiltration',
            'direct_soil_moisture', 'reference_et0', 'actual_et',
            'actual_et_interception', 'crop_etc', 'interception',
            'interception_storage', 'interception_storage_max',
            'tmin', 'tmax', 'tmean', 'crop_coefficient', 'rooting_depth',
            'wilting_point', 'field_capacity', 'available_water_content',
            'landuse_indices', 'soil_indices', 'flow_direction',
            'flow_sequence', 'routing_fraction', 'active'
        ]

        for name in array_names:
            if hasattr(self, name):
                setattr(self, name, None)

    def get_current_state(self) -> Dict[str, NDArray]:
        """Get current state of model variables"""
        return {
            'soil_storage': self.soil_storage,
            'surface_storage': self.surface_storage,
            'snow_storage': self.snow_storage,
            'interception_storage': self.interception_storage,
            'gross_precip': self.gross_precip,
            'rainfall': self.rainfall,
            'snowfall': self.snowfall,
            'runoff': self.runoff,
            'infiltration': self.infiltration,
            'reference_et0': self.reference_et0,
            'actual_et': self.actual_et,
            'tmin': self.tmin,
            'tmax': self.tmax,
            'tmean': self.tmean,
            'crop_coefficient': self.crop_coefficient,
            'rooting_depth': self.rooting_depth,
            'net_infiltration': self.net_infiltration,
            'direct_soil_moisture': self.direct_soil_moisture
        }

    @property
    def is_ready(self) -> bool:
        """Check if model is ready for simulation"""
        return (self.is_initialized and
                self.parameters_initialized and
                self.output_initialized and
                self.start_date is not None and
                self.end_date is not None)

    def validate_state(self) -> None:
        """Validate model state arrays for consistency"""
        if not self.is_initialized:
            raise RuntimeError("Model must be initialized before validation")

        # Check array sizes
        expected_size = self.domain_size
        array_sizes = {
            name: getattr(self, name).size
            for name in dir(self)
            if isinstance(getattr(self, name), np.ndarray)
        }

        mismatched = {
            name: size for name, size in array_sizes.items()
            if size != expected_size
        }

        if mismatched:
            raise ValueError(
                f"Array size mismatch. Expected {expected_size}, "
                f"but found: {mismatched}"
            )

    def reset_state(self) -> None:
        """Reset model state arrays to initial values"""
        self.logger.info("Resetting model state")

        # Re-initialize arrays
        self._initialize_arrays()
        self._initialize_domain_arrays()

        # Reset modules if they exist
        if hasattr(self, 'agriculture_module') and self.agriculture_module is not None:
            self.agriculture_module.reset()
        if hasattr(self, 'weather_module') and self.weather_module is not None:
            self.weather_module.reset()

        # Reset date
        self.current_date = self.start_date