from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import logging

from .runoff import RunoffModule, RunoffParameters
from .agriculture import AgricultureModule, CropParameters
from .actual_et import ActualETCalculator
from .interception import InterceptionModule, GashParameters
from .infiltration import InfiltrationModule, InfiltrationParameters
from .soil import SoilModule, SoilParameters
from .weather import WeatherModule
from .potential_et import PotentialETCalculator
from .logging_config import ParameterLogger


class ModelDomain:
    """
    Core model domain class handling state, calculations, and module coordination.
    Implements optimized integration pattern with proper initialization and state management.
    """

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

        # Time tracking
        self.current_date: Optional[datetime] = None
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None

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

        # Initialize empty arrays (sized during grid initialization)
        self._initialize_arrays()

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

        # Initialize domain size and arrays
        self.domain_size = nx * ny
        self.active = np.ones((ny, nx), dtype=bool)

        # Size arrays
        self._initialize_domain_arrays()

        # Initialize modules with domain size
        self._initialize_modules()

        self.is_initialized = True
        self.logger.info(f"Grid initialized: {nx}x{ny} cells")

    def initialize_parameters(self, landuse_params: Dict[int, Dict],
                              soil_params: Dict[int, Dict]) -> None:
        """Initialize parameters for all modules with validation"""
        if not self.is_initialized:
            raise RuntimeError("Grid must be initialized before parameters")

        self.logger.info("Initializing model parameters")

        try:
            # Validate parameters
            self._validate_landuse_parameters(landuse_params)
            self._validate_soil_parameters(soil_params)

            # Initialize module parameters
            for landuse_id, params in landuse_params.items():
                self.runoff_module.add_parameters(
                    landuse_id, RunoffParameters(**params['runoff']))

                self.agriculture_module.add_crop_parameters(
                    landuse_id, CropParameters(**params['crop']))

                self.interception_module.add_gash_parameters(
                    landuse_id, GashParameters(**params['interception']))

            for soil_id, params in soil_params.items():
                self.infiltration_module.add_soil_parameters(
                    soil_id, InfiltrationParameters(**params['infiltration']))

                self.soil_module.add_soil_parameters(
                    soil_id, SoilParameters(**params['soil']))

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
            # Initialize each module
            self.runoff_module.initialize(landuse_indices)
            self.agriculture_module.initialize(landuse_indices)
            self.soil_module.initialize(soil_indices, self.soil_storage_max)
            self.infiltration_module.initialize(soil_indices, self.soil_storage_max)

            # Initialize weather module
            self.weather_module.initialize(
                fragments_file=fragments_file,
                rainfall_zones=np.ones_like(landuse_indices),
                zones=np.ones_like(landuse_indices),
                monthly_ratios={1: [1.0] * 12},
                elevation=elevation,
                latitude=latitude
            )

            self.logger.info("Module initialization complete")

        except Exception as e:
            self.logger.error(f"Module initialization failed: {str(e)}")
            raise

    def initialize_simulation_period(self, start_date: datetime, end_date: datetime) -> None:
        """Initialize simulation start and end dates"""
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date

        self.logger.info(f"Simulation period: {start_date} to {end_date}")

    def update_date(self, date: datetime) -> None:
        """Update current simulation date and related state"""
        self.current_date = date

        # Update date in modules that need it
        if self.agriculture_module is not None:
            self.agriculture_module.current_date = date

        # Update any time-dependent parameters
        self._update_time_dependent_parameters()

    def get_weather_data(self, date: datetime) -> None:
        """Update weather data for given date"""
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

    def cleanup(self) -> None:
        """Clean up model resources"""
        self.logger.info("Cleaning up model resources")

        try:
            # Clean up modules
            modules = [
                'runoff_module',
                'agriculture_module',
                'actual_et_calculator',
                'interception_module',
                'infiltration_module',
                'soil_module',
                'weather_module',
                'potential_et_calculator'
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

    def _initialize_arrays(self) -> None:
        """Initialize empty arrays"""
        # These will be properly sized during grid initialization
        self.surface_storage = None
        self.snow_storage = None
        self.interception_storage = None
        self.soil_storage = None
        self.soil_storage_max = None
        self.gross_precip = None
        self.rainfall = None
        self.snowfall = None
        self.runoff = None
        self.infiltration = None
        self.reference_et0 = None
        self.actual_et = None
        self.tmin = None
        self.tmax = None
        self.tmean = None
        self.crop_coefficient = None
        self.rooting_depth = None

    def _initialize_domain_arrays(self) -> None:
        """Initialize domain arrays to proper size"""
        n_active = np.count_nonzero(self.active)

        # Initialize all arrays to proper size
        self.surface_storage = np.zeros(n_active, dtype=np.float32)
        self.snow_storage = np.zeros(n_active, dtype=np.float32)
        self.interception_storage = np.zeros(n_active, dtype=np.float32)
        self.soil_storage = np.zeros(n_active, dtype=np.float32)
        self.soil_storage_max = np.zeros(n_active, dtype=np.float32)
        self.gross_precip = np.zeros(n_active, dtype=np.float32)
        self.rainfall = np.zeros(n_active, dtype=np.float32)
        self.snowfall = np.zeros(n_active, dtype=np.float32)
        self.runoff = np.zeros(n_active, dtype=np.float32)
        self.infiltration = np.zeros(n_active, dtype=np.float32)
        self.reference_et0 = np.zeros(n_active, dtype=np.float32)
        self.actual_et = np.zeros(n_active, dtype=np.float32)
        self.tmin = np.zeros(n_active, dtype=np.float32)
        self.tmax = np.zeros(n_active, dtype=np.float32)
        self.tmean = np.zeros(n_active, dtype=np.float32)
        self.crop_coefficient = np.ones(n_active, dtype=np.float32)
        self.rooting_depth = np.zeros(n_active, dtype=np.float32)

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

    def _validate_landuse_parameters(self, params: Dict[int, Dict]) -> None:
        """Validate landuse parameters"""
        required_modules = ['runoff', 'crop', 'interception']
        for landuse_id, landuse_params in params.items():
            missing = [m for m in required_modules if m not in landuse_params]
            if missing:
                raise ValueError(
                    f"Missing required parameter modules for landuse {landuse_id}: {missing}"
                )

    def _validate_soil_parameters(self, params: Dict[int, Dict]) -> None:
        """Validate soil parameters"""
        required_modules = ['infiltration', 'soil']
        for soil_id, soil_params in params.items():
            missing = [m for m in required_modules if m not in soil_params]
            if missing:
                raise ValueError(
                    f"Missing required parameter modules for soil {soil_id}: {missing}"
                )

    def _update_time_dependent_parameters(self) -> None:
        """Update time-dependent parameters"""
        if self.current_date is None:
            return

        if self.agriculture_module is not None:
            self.agriculture_module.update_growing_season(self.tmean)

        if self.current_date.day == 1:
            self._update_monthly_parameters()

    def _update_monthly_parameters(self) -> None:
        """Update any monthly parameters"""
        # Implementation depends on which parameters vary monthly
        pass

    def _create_flow_sequence(self) -> None:
        """Create sequence of cell indices from upslope to downslope"""
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
        """Recursively trace flow path from a cell"""
        # Mark current cell as visited
        visited[cell] = True

        # Get next downslope cell
        next_cell = self.flow_direction[cell]

        # Follow flow path if valid cell and not already visited
        if next_cell >= 0 and not visited[next_cell]:
            self._trace_flow_path(next_cell, visited, sequence)

        # Add current cell to sequence
        sequence.append(cell)

    def _clear_arrays(self) -> None:
        """Clear memory by setting arrays to None"""
        array_attrs = [
            'surface_storage',
            'snow_storage',
            'interception_storage',
            'soil_storage',
            'soil_storage_max',
            'gross_precip',
            'rainfall',
            'snowfall',
            'runoff',
            'infiltration',
            'reference_et0',
            'actual_et',
            'tmin',
            'tmax',
            'tmean',
            'crop_coefficient',
            'rooting_depth'
        ]

        for attr in array_attrs:
            if hasattr(self, attr):
                setattr(self, attr, None)

    def initialize_output(self, directory: Path, prefix: str,
                          variables: List[str], compression: bool = True) -> None:
        """Initialize output configuration"""
        if not self.is_initialized:
            raise RuntimeError("Domain must be initialized before configuring output")

        self.output_dir = directory
        directory.mkdir(parents=True, exist_ok=True)

        self.output_prefix = prefix
        self.output_variables = variables
        self.output_compression = compression

        # Initialize output metadata for each variable
        self.output_metadata = {}
        for var in variables:
            if hasattr(self, var.lower()):
                self.output_metadata[var] = {
                    'units': self._get_variable_units(var),
                    'description': self._get_variable_description(var),
                    'missing_value': -9999.0
                }

        self.output_initialized = True
        self.logger.info("Output configuration initialized")

    def _get_variable_units(self, variable: str) -> str:
        """Get units for a given variable"""
        units_map = {
            'precipitation': 'inches',
            'rainfall': 'inches',
            'snowfall': 'inches',
            'runoff': 'inches',
            'infiltration': 'inches',
            'soil_storage': 'inches',
            'reference_et0': 'inches',
            'actual_et': 'inches',
            'tmin': 'degrees F',
            'tmax': 'degrees F',
            'tmean': 'degrees F',
            'crop_coefficient': 'dimensionless',
            'rooting_depth': 'inches'
        }
        return units_map.get(variable.lower(), 'unknown')

    def _get_variable_description(self, variable: str) -> str:
        """Get description for a given variable"""
        description_map = {
            'precipitation': 'Total precipitation',
            'rainfall': 'Rainfall component of precipitation',
            'snowfall': 'Snowfall component of precipitation',
            'runoff': 'Surface water runoff',
            'infiltration': 'Water infiltration into soil',
            'soil_storage': 'Soil water storage',
            'reference_et0': 'Reference evapotranspiration',
            'actual_et': 'Actual evapotranspiration',
            'tmin': 'Minimum daily temperature',
            'tmax': 'Maximum daily temperature',
            'tmean': 'Mean daily temperature',
            'crop_coefficient': 'Crop coefficient',
            'rooting_depth': 'Root zone depth'
        }
        return description_map.get(variable.lower(), 'No description available')

    def calc_reference_et(self) -> None:
        """Calculate reference evapotranspiration"""
        if self.current_date is None:
            raise RuntimeError("Current date must be set before calculating reference ET")

        self.reference_et0 = self.potential_et_calculator.calculate(
            date=self.current_date,
            tmin=self.tmin,
            tmax=self.tmax
        )

    def process_timestep(self, date: datetime) -> None:
        """Process model state for current timestep"""
        if not self.is_initialized:
            raise RuntimeError("Model must be initialized before processing timesteps")

        # Update date
        self.update_date(date)

        # Get weather data
        self.get_weather_data(date)

        # Calculate reference ET
        self.calc_reference_et()

        # Update crop coefficients if using FAO-56
        if self.use_crop_coefficients:
            self.agriculture_module.update_crop_coefficients()

        # Update rooting depth if using dynamic rooting
        if self.dynamic_rooting:
            self.agriculture_module.update_root_depth()

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
            'rooting_depth': self.rooting_depth
        }

    @property
    def is_ready(self) -> bool:
        """Check if model is ready for simulation"""
        return (self.is_initialized and
                self.parameters_initialized and
                self.output_initialized and
                self.start_date is not None and
                self.end_date is not None)