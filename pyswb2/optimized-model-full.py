from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Any
import numpy as np
import netCDF4 as nc
from dataclasses import dataclass
import logging

from .configuration import ConfigurationManager, ModelConfig
from .grid import Grid, GridDataType
from .daily_calculation import DailyCalculation
from .logging_config import ParameterLogger
from .grid_clipper import GridClipper, GridExtent
from .runoff_diagnostics import RunoffDiagnostics
from .runoff import RunoffModule
from .agriculture import AgricultureModule
from .actual_et import ActualETCalculator
from .interception import InterceptionModule
from .infiltration import InfiltrationModule
from .soil import SoilModule
from .weather import WeatherModule
from .potential_et import PotentialETCalculator

class CombinedModel:
    """Unified model class combining runner and domain functionality"""
    
    def __init__(self, config_path: Path, log_dir: Optional[Path] = None):
        # Initialize logging
        self.logger = ParameterLogger(log_dir)
        self.logger.info(f"Initializing combined model with config: {config_path}")
        
        # Core configuration
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.load(config_path)
        
        # Initialize components
        self.daily_calc = DailyCalculation(self)
        self.runoff_diagnostics = RunoffDiagnostics(self.logger)
        
        # Initialize grid clipper
        target_extent = GridExtent(
            xmin=self.config.grid.x0,
            ymin=self.config.grid.y0,
            xmax=self.config.grid.x0 + self.config.grid.nx * self.config.grid.cell_size,
            ymax=self.config.grid.y0 + self.config.grid.ny * self.config.grid.cell_size,
            cell_size=self.config.grid.cell_size,
            nx=self.config.grid.nx,
            ny=self.config.grid.ny
        )
        self.grid_clipper = GridClipper(target_extent)
        
        # Initialize state variables
        self._init_state_variables()
        
        # Initialize module instances
        self._init_modules()
        
        self.is_initialized = False
        
    def _init_state_variables(self):
        """Initialize all state tracking variables"""
        # Grid properties
        self.number_of_columns = 0
        self.number_of_rows = 0
        self.gridcellsize = 0.0
        self.x_ll = 0.0
        self.y_ll = 0.0
        self.proj4_string = ""
        self.domain_size = 0
        self.active = None
        
        # Time tracking
        self.current_date = None
        self.start_date = None
        self.end_date = None
        
        # Initialize arrays
        self._init_domain_arrays()
        
        # Configuration flags
        self.routing_enabled = False
        self.dynamic_rooting = False
        self.use_crop_coefficients = False
        
        # Routing variables
        self.flow_direction = None
        self.flow_sequence = None
        self.routing_fraction = None
        
        # Module indices
        self.landuse_indices = None
        self.soil_indices = None
        self.elevation = None
        self.latitude = None
        
        # Output tracking
        self.output_files = {}
        self.output_paths = {}
        
    def _init_domain_arrays(self):
        """Initialize all domain arrays"""
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

    def _init_modules(self):
        """Initialize all model module instances"""
        self.domain_size = self.config.grid.nx * self.config.grid.ny
        
        self.runoff_module = RunoffModule(self.domain_size)
        self.agriculture_module = AgricultureModule(self.domain_size)
        self.actual_et_calculator = ActualETCalculator(self.domain_size)
        self.interception_module = InterceptionModule(self.domain_size)
        self.infiltration_module = InfiltrationModule(self.domain_size)
        self.soil_module = SoilModule(self.domain_size)
        self.weather_module = WeatherModule(self.domain_size, (self.number_of_rows, self.number_of_columns))
        self.potential_et_calculator = PotentialETCalculator(self.domain_size)
        
    def initialize(self):
        """Initialize model in correct order"""
        try:
            self._load_input_data()
            self._initialize_grid()
            self._initialize_domain()
            self._initialize_output()
            
            self.is_initialized = True
            self.logger.info("Model initialization complete")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise
            
    def _load_input_data(self):
        """Load and align required input data"""
        self.logger.info("Loading input data")
        
        try:
            # Load landuse and soils grids
            landuse_grid = self._load_grid(self.config.input.landuse_grid)
            soils_grid = self._load_grid(self.config.input.hydrologic_soils_group)
            
            # Convert to indices
            self.landuse_indices = landuse_grid.astype(np.int32)
            self.soil_indices = soils_grid.astype(np.int32)
            
            # Create default elevation and latitude arrays
            grid_shape = (self.config.grid.ny, self.config.grid.nx)
            self.elevation = np.zeros(grid_shape, dtype=np.float32)
            self.latitude = np.full(grid_shape, 20.7, dtype=np.float32)
            
            if self.config.routing_enabled:
                self.flow_direction = self._load_grid(self.config.flow_direction_grid)
                self.routing_fraction = np.ones_like(self.flow_direction, dtype=np.float32)
                
        except Exception as e:
            self.logger.error(f"Failed to load input data: {str(e)}")
            raise

    def _load_grid(self, path: Path) -> np.ndarray:
        """Load and align grid data"""
        self.logger.info(f"Loading and aligning grid: {path}")
        
        try:
            output_path = path.parent / f"aligned_{path.name}"
            
            self.grid_clipper.process_file(
                input_path=path,
                output_path=output_path,
                method='nearest',
                fill_value=-9999
            )
            
            data, metadata = self.grid_clipper.read_grid(output_path)
            
            if data.shape != (self.config.grid.ny, self.config.grid.nx):
                raise ValueError(f"Grid shape {data.shape} doesn't match expected "
                             f"({self.config.grid.ny}, {self.config.grid.nx})")
                             
            if output_path.exists():
                output_path.unlink()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load and align grid {path}: {str(e)}")
            raise

    def _initialize_grid(self):
        """Initialize model grid"""
        self.logger.info("Initializing model grid")
        
        if self.config.grid.nx <= 0 or self.config.grid.ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.config.grid.cell_size <= 0:
            raise ValueError("Cell size must be positive")
            
        self.number_of_columns = self.config.grid.nx
        self.number_of_rows = self.config.grid.ny
        self.gridcellsize = self.config.grid.cell_size
        self.x_ll = self.config.grid.x0
        self.y_ll = self.config.grid.y0
        self.proj4_string = self.config.grid.proj4_string
        
        self.domain_size = self.number_of_columns * self.number_of_rows
        self.active = np.ones((self.number_of_rows, self.number_of_columns), dtype=bool)
        
        n_active = np.count_nonzero(self.active)
        
        # Initialize arrays
        for name in [attr for attr in dir(self) if isinstance(getattr(self, attr), np.ndarray)]:
            if name == 'crop_coefficient':
                setattr(self, name, np.ones(n_active, dtype=np.float32))
            else:
                setattr(self, name, np.zeros(n_active, dtype=np.float32))

    def _initialize_domain(self):
        """Initialize model domain components"""
        if not self.is_initialized:
            raise RuntimeError("Grid must be initialized before domain")
            
        # Process parameters
        landuse_params = self._process_landuse_parameters()
        soil_params = self._process_soil_parameters()
        
        # Initialize modules
        self.runoff_module.initialize(self.landuse_indices.ravel())
        self.agriculture_module.initialize(self.landuse_indices.ravel())
        self.soil_module.initialize(self.soil_indices.ravel(), self.soil_storage_max)
        self.infiltration_module.initialize(self.soil_indices.ravel(), self.soil_storage_max)
        
        self.interception_module.initialize(
            landuse_indices=self.landuse_indices.ravel(),
            canopy_cover=np.ones_like(self.landuse_indices.ravel(), dtype=np.float32),
            evap_to_rain_ratio=np.ones_like(self.landuse_indices.ravel(), dtype=np.float32)
        )
        
        self.weather_module.initialize(
            fragments_file=self.config.input.fragments_file,
            rainfall_zones=np.ones_like(self.landuse_indices),
            zones=np.ones_like(self.landuse_indices),
            monthly_ratios={1: [1.0] * 12},
            elevation=self.elevation.ravel(),
            latitude=self.latitude.ravel()
        )
        
        self.potential_et_calculator.latitude = self.latitude.ravel()
        
        self.actual_et_calculator.initialize(
            landuse_indices=self.landuse_indices.ravel(),
            soil_indices=self.soil_indices.ravel(),
            elevation=self.elevation.ravel()
        )
        
        # Initialize simulation period
        self.start_date = self.config.start_date
        self.end_date = self.config.end_date
        self.current_date = self.start_date

    def _initialize_output(self):
        """Initialize model output"""
        self.logger.info("Initializing output configuration")
        
        self.config.output.directory.mkdir(parents=True, exist_ok=True)
        
        total_days = (self.config.end_date - self.config.start_date).days + 1
        
        for var in self.config.output.variables:
            clean_var = var.lstrip('_')
            
            output_path = (self.config.output.directory /
                           f"{self.config.output.prefix}{clean_var}.nc")
            self.output_paths[var] = output_path
            self._initialize_variable_netcdf(var, output_path, total_days)

    def _initialize_variable_netcdf(self, variable: str, output_path: Path,
                                  total_timesteps: int):
        """Initialize NetCDF file for a variable"""
        try:
            ds = nc.Dataset(output_path, 'w', format='NETCDF4')
            
            # Create dimensions
            ds.createDimension('y', self.config.grid.ny)
            ds.createDimension('x', self.config.grid.nx)
            ds.createDimension('time', total_timesteps)
            
            # Create coordinate variables
            x = ds.createVariable('x', 'f8', ('x',))
            y = ds.createVariable('y', 'f8', ('y',))
            time = ds.createVariable('time', 'f8', ('time',))
            
            # Set coordinate values
            x[:] = np.linspace(
                self.config.grid.x0,
                self.config.grid.x0 + self.config.grid.nx * self.config.grid.cell_size,
                self.config.grid.nx
            )
            y[:] = np.linspace(
                self.config.grid.y0,
                self.config.grid.y0 + self.config.grid.ny * self.config.grid.cell_size,
                self.config.grid.ny
            )
            
            time[:] = np.arange(total_timesteps)
            time.units = f'days since {self.config.start_date.strftime("%Y-%m-%d")}'
            time.calendar = 'standard'
            
            # Get clean variable name and metadata
            clean_var = variable.lstrip('_')
            metadata = self._get_output_metadata().get(clean_var.lower(), {})
            
            # Create data variable
            v = ds.createVariable(
                clean_var,
                'f4',
                ('time', 'y', 'x'),
                zlib=self.config.output.compression,
                fill_value=-9999.0,
                least_significant_digit=3
            )
            
            # Add metadata
            for key, value in metadata.items():
                setattr(v, key, value)
                
            # Initialize with fill values
            v[:] = v._FillValue
            
            self.output_files[variable] = ds
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NetCDF file for {variable}: {str(e)}")
            raise

    def run(self):
        """Run model simulation"""
        if not self.is_initialized:
            raise RuntimeError("Model must be initialized before running")
            
        self.logger.info(f"Starting simulation from {self.config.start_date} "
                        f"to {self.config.end_date}")
                        
        try:
            current_date = self.config.start_date
            while current_date <= self.config.end_date:
                self._process_timestep(current_date)
                current_date += timedelta(days=1)

            self.logger.info("Simulation completed successfully")

        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            raise
        finally:
            self.cleanup()

    def _process_timestep(self, date: datetime):
        """Process single timestep"""
        self.logger.debug(f"Processing timestep: {date}")

        # Update model date
        self.current_date = date

        # Get weather data
        self.get_weather_data(date)

        # Collect and log diagnostics
        diagnostics = self.runoff_diagnostics.collect_diagnostics(self, date)
        self.runoff_diagnostics.log_diagnostics(diagnostics)

        # Run daily calculations
        self.daily_calc.perform_daily_calculation(date)

        # Write output if configured
        if self.config.output.write_daily:
            self._write_output(date)

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

    def _write_output(self, date: datetime) -> None:
        """Write output for current timestep"""
        try:
            time_index = (date - self.config.start_date).days
            state = self.get_current_state()

            for var in self.config.output.variables:
                if var in state:
                    try:
                        ds = self.output_files[var]
                        if ds is None or not isinstance(ds, nc.Dataset):
                            ds = nc.Dataset(self.output_paths[var], 'a')
                            self.output_files[var] = ds

                        clean_var = var.lstrip('_')
                        grid_data = state[var].reshape(self.config.grid.ny, self.config.grid.nx)
                        ds.variables[clean_var][time_index, :, :] = grid_data
                        ds.sync()

                    except Exception as e:
                        self.logger.error(f"Error writing variable {var} at time {date}: {str(e)}")
                        raise

        except Exception as e:
            self.logger.error(f"Failed to write output for {date}: {str(e)}")
            raise

    def get_current_state(self) -> Dict[str, np.ndarray]:
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

    def _process_landuse_parameters(self) -> Dict[int, Dict]:
        """Process landuse parameters from lookup table"""
        self.logger.info("Processing landuse parameters")

        landuse_params = {}
        lookup_path = self.config.input.lookup_tables.get('landuse')
        if not lookup_path:
            raise ValueError("Landuse lookup table not found in configuration")

        with open(lookup_path) as f:
            header = None
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', '!', '$', '%')):
                    continue

                if header is None:
                    header = [col.strip() for col in line.split('\t')]
                    continue

                values = line.split('\t')
                if len(values) != len(header):
                    continue

                params = dict(zip(header, values))
                try:
                    lu_code = int(params['LU code'])
                    landuse_params[lu_code] = self._parse_landuse_parameters(params)
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error processing landuse code: {e}")

        return landuse_params

    def _parse_landuse_parameters(self, params: Dict) -> Dict:
        """Parse individual landuse parameters"""
        parsed = {
            'runoff': {
                'curve_number': float(params.get('CN 2', 70.0)),
                'initial_abstraction_ratio': 0.2,
                'depression_storage': float(params.get('Surface Storage Max', 0.0)),
                'impervious_fraction': float(params['Assumed Imperviousness'])
                if params.get('Assumed Imperviousness', '**') != '**' else 0.0
            },
            'crop': {
                'gdd_base': float(params.get('GDD_Base_Temperature', 50.0)),
                'gdd_max': float(params.get('GDD_Maximum_Temperature', 86.0)),
                'growing_season_start': params.get('First_day_of_growing_season'),
                'growing_season_end': params.get('Last_day_of_growing_season'),
                'initial_root_depth': float(params.get('Rooting_depth_inches', 0.0)),
                'max_root_depth': float(params.get('Rooting_depth_inches', 0.0)),
                'crop_coefficient': float(params.get('Kcb_mid', 1.0))
            },
            'interception': {
                'canopy_storage_capacity': float(params.get('Canopy_Storage_Capacity', 0.0)),
                'trunk_storage_capacity': float(params.get('Trunk_Storage_Capacity', 0.0)),
                'stemflow_fraction': float(params.get('Stemflow_Fraction', 0.0)),
                'interception_storage_max_growing': float(params.get('Interception_Growing', 0.1)),
                'interception_storage_max_nongrowing': float(params.get('Interception_Nongrowing', 0.1))
            }
        }

        # Add irrigation if available
        if 'Max_allowable_depletion' in params or 'Irrigation_efficiency' in params:
            parsed['irrigation'] = {
                'max_allowable_depletion': float(params.get('Max_allowable_depletion', 0.0)),
                'efficiency': float(params.get('Irrigation_efficiency', 1.0))
            }

        return parsed

    def _process_soil_parameters(self) -> Dict[int, Dict]:
        """Process soil parameters from lookup table"""
        return {
            1: {
                'infiltration': {
                    'maximum_rate': 2.0,
                    'minimum_rate': 0.0,
                    'soil_storage_max': 8.0
                },
                'soil': {
                    'awc': 0.2,
                    'field_capacity': 0.3,
                    'wilting_point': 0.1,
                    'hydraulic_conductivity': 1.0
                }
            }
        }

    def cleanup(self):
        """Clean up model resources"""
        self.logger.info("Cleaning up model resources")

        try:
            # Close output files
            for ds in self.output_files.values():
                if ds is not None and isinstance(ds, nc.Dataset):
                    try:
                        ds.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing NetCDF file: {str(e)}")
            self.output_files.clear()

            # Clean up modules
            self._cleanup_modules()

            # Clear arrays
            self._clear_arrays()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

    def _cleanup_modules(self):
        """Clean up all module resources"""
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

    def _clear_arrays(self):
        """Clear all numpy arrays"""
        array_attrs = [
            attr for attr, value in vars(self).items()
            if isinstance(value, np.ndarray)
        ]

        for attr in array_attrs:
            setattr(self, attr, None)

    def _get_output_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for output variables"""
        return {
            'gross_precipitation': {
                'units': 'inches',
                'long_name': 'Gross Precipitation',
                'description': 'Total precipitation before interception'
            },
            'rainfall': {
                'units': 'inches',
                'long_name': 'Rainfall',
                'description': 'Liquid precipitation'
            },
            'snowfall': {
                'units': 'inches',
                'long_name': 'Snowfall',
                'description': 'Frozen precipitation'
            },
            # Add other variable metadata as needed
        }

    def run_swb_model(config_path: Path, log_dir: Optional[Path] = None) -> None:
        """Convenience function to run SWB model"""
        model = CombinedModel(config_path, log_dir)
        try:
            model.initialize()
            model.run()
        except Exception as e:
            logging.error(f"Model run failed: {str(e)}")
            raise