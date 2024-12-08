from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Any, Generator
import numpy as np
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import netCDF4 as nc
from .configuration import ConfigurationManager, ModelConfig
from .model_domain import ModelDomain
from .grid import Grid, GridDataType
from .daily_calculation import DailyCalculation
from .logging_config import ParameterLogger
from .grid_clipper import GridClipper, GridExtent, GridFormat  # Import the existing utility


class SWBModelRunner:
    """Main runner class for SWB model with input validation"""

    def __init__(self, config_path: Path, log_dir: Optional[Path] = None):
        """Initialize model components"""
        # Set up logging first
        self.logger = ParameterLogger(log_dir)
        self.logger.info(f"Initializing SWB model with config: {config_path}")

        # Core components
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.load(config_path)
        self.domain = ModelDomain(log_dir)
        self.daily_calc = DailyCalculation(self.domain)

        # Initialize grid clipper with target grid specifications
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

        # Initialize arrays
        self.landuse_indices = None
        self.soil_indices = None
        self.elevation = None
        self.latitude = None
        self.flow_direction = None

        # State tracking
        self.is_initialized = False

    def _load_grid(self, path: Path) -> np.ndarray:
        """Load and align grid data using grid clipper utility"""
        self.logger.info(f"Loading and aligning grid: {path}")

        try:
            # Process the file using grid clipper
            output_path = path.parent / f"aligned_{path.name}"
            self.grid_clipper.process_file(
                input_path=path,
                output_path=output_path,
                method='nearest',  # Use nearest neighbor for categorical data
                fill_value=-9999
            )

            # Read the aligned data
            data, metadata = self.grid_clipper.read_grid(output_path)

            # Verify grid shape
            if data.shape != (self.config.grid.ny, self.config.grid.nx):
                raise ValueError(f"Grid shape {data.shape} doesn't match expected "
                                 f"({self.config.grid.ny}, {self.config.grid.nx})")

            # Clean up temporary file if needed
            if output_path.exists():
                output_path.unlink()

            return data

        except Exception as e:
            self.logger.error(f"Failed to load and align grid {path}: {str(e)}")
            raise

    def _load_input_data(self) -> None:
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

            # Load optional grids
            if self.config.routing_enabled:
                self.flow_direction = self._load_grid(self.config.flow_direction_grid)
                self.routing_fraction = np.ones_like(self.flow_direction, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Failed to load input data: {str(e)}")
            raise

    def initialize(self) -> None:
        """Initialize model in correct order"""
        try:
            # 1. Load input data with alignment
            self._load_input_data()

            # 2. Initialize grid
            self._initialize_grid()

            # 3. Initialize domain with parameters and data
            self._initialize_domain()

            # 4. Initialize output
            self._initialize_output()

            self.is_initialized = True
            self.logger.info("Model initialization complete")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_grid(self):
        """Initialize model grid from configuration"""
        self.logger.info("Initializing model grid")

        self.domain.initialize_grid(
            nx=self.config.grid.nx,
            ny=self.config.grid.ny,
            x_ll=self.config.grid.x0,
            y_ll=self.config.grid.y0,
            cell_size=self.config.grid.cell_size,
            proj4=self.config.grid.proj4_string
        )

    def _initialize_domain(self) -> None:
        """Initialize model domain components"""
        # Process parameters
        landuse_params = self._process_landuse_parameters()
        soil_params = self._process_soil_parameters()

        # Initialize domain with parameters
        self.domain.initialize_parameters(landuse_params, soil_params)

        # Ensure arrays are properly flattened before passing to modules
        landuse_indices_flat = self.landuse_indices.ravel()
        soil_indices_flat = self.soil_indices.ravel()
        elevation_flat = self.elevation.ravel()
        latitude_flat = self.latitude.ravel()

        # Verify sizes
        domain_size = self.config.grid.nx * self.config.grid.ny
        self.logger.info(f"Verifying array sizes (should be {domain_size}):")
        self.logger.info(f"Landuse indices: {len(landuse_indices_flat)}")
        self.logger.info(f"Soil indices: {len(soil_indices_flat)}")
        self.logger.info(f"Elevation: {len(elevation_flat)}")
        self.logger.info(f"Latitude: {len(latitude_flat)}")

        # Initialize modules with flattened data
        self.domain.initialize_modules(
            landuse_indices=landuse_indices_flat,
            soil_indices=soil_indices_flat,
            elevation=elevation_flat,
            latitude=latitude_flat,
            fragments_file=self.config.input.fragments_file
        )

        # Initialize simulation period
        self.domain.initialize_simulation_period(
            self.config.start_date,
            self.config.end_date
        )

    def _process_landuse_parameters(self) -> Dict[int, Dict]:
        """Process landuse parameters from lookup table"""
        self.logger.info("Processing landuse parameters")

        landuse_params = {}
        lookup_path = self.config.input.lookup_tables.get('landuse')
        if not lookup_path:
            raise ValueError("Landuse lookup table not found in configuration")

        # Read lookup table
        with open(lookup_path) as f:
            # Skip initial comments
            header = None
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', '!', '$', '%')):
                    continue

                if header is None:
                    # Parse header
                    header = [col.strip() for col in line.split('\t')]
                    continue

                # Parse parameter values
                values = line.split('\t')
                if len(values) != len(header):
                    continue  # Skip malformed lines

                # Create parameter dictionary
                params = dict(zip(header, values))

                try:
                    lu_code = int(params['LU code'])

                    landuse_params[lu_code] = {
                        'runoff': {
                            'curve_number': float(params.get('CN 2', 70.0)),
                            'initial_abstraction_ratio': 0.2,
                            'depression_storage': float(params.get('Surface Storage Max', 0.0)),
                            'impervious_fraction': float(params['Assumed Imperviousness'])
                            if params.get('Assumed Imperviousness', '**') != '**' else 0.0
                        },
                        'interception': {
                            'canopy_storage_capacity': float(params.get('Canopy_Storage_Capacity', 0.0)),
                            'trunk_storage_capacity': float(params.get('Trunk_Storage_Capacity', 0.0)),
                            'stemflow_fraction': float(params.get('Stemflow_Fraction', 0.0)),
                            'interception_storage_max_growing': float(params.get('Interception_Growing', 0.1)),
                            'interception_storage_max_nongrowing': float(params.get('Interception_Nongrowing', 0.1))
                        },
                        'crop': {
                            'gdd_base': float(params.get('GDD_Base_Temperature', 50.0)),
                            'gdd_max': float(params.get('GDD_Maximum_Temperature', 86.0)),
                            'growing_season_start': params.get('First_day_of_growing_season'),
                            'growing_season_end': params.get('Last_day_of_growing_season'),
                            'initial_root_depth': float(params.get('Rooting_depth_inches', 0.0)),
                            'max_root_depth': float(params.get('Rooting_depth_inches', 0.0)),
                            'crop_coefficient': float(params.get('Kcb_mid', 1.0))
                        }
                    }

                    # Add irrigation if available
                    irrigation_params = {}
                    if 'Max_allowable_depletion' in params:
                        irrigation_params['max_allowable_depletion'] = float(params['Max_allowable_depletion'])
                    if 'Irrigation_efficiency' in params:
                        irrigation_params['efficiency'] = float(params['Irrigation_efficiency'])
                    if irrigation_params:
                        landuse_params[lu_code]['irrigation'] = irrigation_params

                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error processing landuse code {lu_code}: {e}")
                    continue

        return landuse_params

    def _process_soil_parameters(self) -> Dict[int, Dict]:
        """Process soil parameters from lookup table"""
        self.logger.info("Processing soil parameters")

        # For now, return default soil parameters
        # This could be expanded to read from a soil lookup table
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

    def _initialize_output(self):
        """Initialize output configuration"""
        self.logger.info("Initializing output configuration")

        self.domain.initialize_output(
            directory=self.config.output.directory,
            prefix=self.config.output.prefix,
            variables=self.config.output.variables,
            compression=self.config.output.compression
        )

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
        self.domain.update_date(date)

        # Get weather data first
        self.domain.get_weather_data(date)

        # Run daily calculations
        self.daily_calc.perform_daily_calculation(date)

        # Write output if configured
        if self.config.output.write_daily:
            self._write_output(date)

    def _write_output(self, date: datetime):
        """Write output for current timestep"""
        output_path = (self.config.output.directory /
                       f"{self.config.output.prefix}_{date:%Y%m%d}.nc")

        # Get current state
        state = self.domain.get_current_state()

        # Write variables configured for output
        with self._get_output_file(output_path) as ds:
            for var in self.config.output.variables:
                if var in state:
                    ds.variables[var][:] = state[var]

    def _get_output_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all possible output variables"""
        metadata = {
            # Water Balance Components
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
            'interception': {
                'units': 'inches',
                'long_name': 'Canopy Interception',
                'description': 'Precipitation intercepted by canopy'
            },
            'runon': {
                'units': 'inches',
                'long_name': 'Run-on',
                'description': 'Surface water run-on from upslope cells'
            },
            'runoff': {
                'units': 'inches',
                'long_name': 'Surface Runoff',
                'description': 'Surface water runoff'
            },
            'runoff_outside': {
                'units': 'inches',
                'long_name': 'Runoff Outside Domain',
                'description': 'Surface runoff leaving the model domain'
            },

            # Soil Water Components
            'infiltration': {
                'units': 'inches',
                'long_name': 'Infiltration',
                'description': 'Water infiltration into soil'
            },
            'net_infiltration': {
                'units': 'inches',
                'long_name': 'Net Infiltration',
                'description': 'Net water infiltration after losses'
            },
            'rejected_net_infiltration': {
                'units': 'inches',
                'long_name': 'Rejected Net Infiltration',
                'description': 'Infiltration rejected due to soil storage limitations'
            },
            'soil_storage': {
                'units': 'inches',
                'long_name': 'Soil Moisture Storage',
                'description': 'Water stored in soil profile'
            },
            'delta_soil_storage': {
                'units': 'inches',
                'long_name': 'Change in Soil Storage',
                'description': 'Daily change in soil moisture storage'
            },

            # ET Components
            'reference_et0': {
                'units': 'inches',
                'long_name': 'Reference ET',
                'description': 'Reference evapotranspiration'
            },
            'actual_et': {
                'units': 'inches',
                'long_name': 'Actual ET',
                'description': 'Actual evapotranspiration'
            },
            'crop_et': {
                'units': 'inches',
                'long_name': 'Crop ET',
                'description': 'Crop-specific evapotranspiration'
            },

            # Temperature Components
            'tmin': {
                'units': 'degrees F',
                'long_name': 'Minimum Temperature',
                'description': 'Daily minimum temperature'
            },
            'tmax': {
                'units': 'degrees F',
                'long_name': 'Maximum Temperature',
                'description': 'Daily maximum temperature'
            },

            # Snow Components
            'snow_storage': {
                'units': 'inches',
                'long_name': 'Snow Storage',
                'description': 'Water equivalent of stored snow'
            },
            'snowmelt': {
                'units': 'inches',
                'long_name': 'Snowmelt',
                'description': 'Daily snowmelt'
            },

            # Growing Degree Days
            'gdd': {
                'units': 'degrees F',
                'long_name': 'Growing Degree Days',
                'description': 'Accumulated growing degree days'
            },

            # Direct Recharge Components
            'direct_net_infiltation': {  # Note: maintaining original spelling from error
                'units': 'inches',
                'long_name': 'Direct Net Infiltration',
                'description': 'Direct infiltration from specific sources'
            },
            'direct_soil_moisture': {
                'units': 'inches',
                'long_name': 'Direct Soil Moisture',
                'description': 'Direct additions to soil moisture'
            },

            # Irrigation
            'irrigation': {
                'units': 'inches',
                'long_name': 'Irrigation',
                'description': 'Applied irrigation water'
            }
        }
        return metadata

    @contextmanager
    def _get_output_file(self, path: Path) -> Generator[nc.Dataset, None, None]:
        """Create and manage NetCDF output file with error handling"""
        try:
            # Create output directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Open NetCDF file
            ds = nc.Dataset(path, 'w', format='NETCDF4')

            try:
                # Create dimensions
                ds.createDimension('y', self.config.grid.ny)
                ds.createDimension('x', self.config.grid.nx)
                ds.createDimension('time', None)  # Unlimited dimension

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

                # Set time units and attributes
                time.units = f'days since {self.config.start_date.strftime("%Y-%m-%d")}'
                time.calendar = 'standard'

                # Add global attributes
                ds.description = 'SWB Model Output'
                ds.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                ds.source = 'pySWB2 Model'
                ds.proj4 = self.config.grid.proj4_string
                ds.model_version = '2.0'

                # Create variables for configured outputs
                metadata = self._get_output_metadata()
                for var in self.config.output.variables:
                    if var.lower() in metadata:  # Case-insensitive matching
                        var_meta = metadata[var.lower()]
                        v = ds.createVariable(
                            var,
                            'f4',
                            ('time', 'y', 'x'),
                            zlib=self.config.output.compression,
                            fill_value=-9999.0,
                            least_significant_digit=3
                        )
                        # Add variable metadata
                        for key, value in var_meta.items():
                            setattr(v, key, value)
                    else:
                        # Create variable without metadata if not in metadata dictionary
                        self.logger.warning(f"No metadata found for variable {var}, creating with defaults")
                        v = ds.createVariable(
                            var,
                            'f4',
                            ('time', 'y', 'x'),
                            zlib=self.config.output.compression,
                            fill_value=-9999.0
                        )

                yield ds

            finally:
                ds.close()

        except Exception as e:
            self.logger.error(f"Error creating output file {path}: {str(e)}")
            raise

    def _write_output(self, date: datetime) -> None:
        """Write output for current timestep"""
        try:
            # Calculate time index
            time_index = (date - self.config.start_date).days

            # Get output path
            output_path = (self.config.output.directory /
                           f"{self.config.output.prefix}_{date:%Y%m%d}.nc")

            # Get current state
            state = self.domain.get_current_state()

            # Write output
            with self._get_output_file(output_path) as ds:
                # Set time value
                ds.variables['time'][time_index] = time_index

                # Write each configured variable
                for var in self.config.output.variables:
                    if var in state:
                        # Reshape flattened array to grid dimensions
                        grid_data = state[var].reshape(self.config.grid.ny, self.config.grid.nx)
                        ds.variables[var][time_index, :, :] = grid_data

        except Exception as e:
            self.logger.error(f"Failed to write output for {date}: {str(e)}")
            raise

    def cleanup(self):
        """Clean up model resources"""
        self.logger.info("Cleaning up model resources")

        try:
            # Clean up domain
            self.domain.cleanup()

            # Clear data arrays
            self.landuse_indices = None
            self.soil_indices = None
            self.elevation = None
            self.latitude = None
            self.flow_direction = None
            self.routing_fraction = None

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise


def run_swb_model(config_path: Path, log_dir: Optional[Path] = None) -> None:
    """Convenience function to run SWB model"""
    runner = SWBModelRunner(config_path, log_dir)
    try:
        runner.initialize()
        runner.run()
    except Exception as e:
        logging.error(f"Model run failed: {str(e)}")
        raise