from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List, Any
import numpy as np
import netCDF4 as nc
import logging

from .configuration import ConfigurationManager, ModelConfig
from .model_domain import ModelDomain
from .grid import Grid, GridDataType
from .daily_calculation import DailyCalculation
from .data_catalog import DataCatalog
from .parameters import Parameters
from .logging_config import ParameterLogger
from .data_interface import DataManager


class SWBModel:
    """
    Main SWB model class implementing optimized integration pattern with
    clear separation of concerns and efficient data flow.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize SWB model components"""
        # Set up logging first
        self.logger = ParameterLogger(log_dir)
        self.logger.info("Initializing SWB model")

        # Core components
        self.config_manager = ConfigurationManager()
        self.domain = ModelDomain(log_dir)
        self.daily_calc = DailyCalculation(self.domain)
        self.data_manager = DataManager()
        self.parameters = Parameters(log_dir)

        # State tracking
        self.initialized: bool = False
        self.config: Optional[ModelConfig] = None

        # Data arrays
        self.landuse_indices: Optional[np.ndarray] = None
        self.soil_indices: Optional[np.ndarray] = None
        self.elevation: Optional[np.ndarray] = None
        self.latitude: Optional[np.ndarray] = None

        # Output handling
        self._nc_out: Optional[nc.Dataset] = None

    def initialize(self, config_path: Path) -> None:
        """Initialize model with configuration"""
        try:
            # Load and validate configuration
            self.logger.info(f"Loading configuration from {config_path}")
            self.config = self.config_manager.load(config_path)
            self.config.validate()

            # Initialize in correct order
            self._initialize_grid()
            self._initialize_parameters()
            self._load_input_data()
            self._initialize_domain()
            self._initialize_output()

            self.initialized = True
            self.logger.info("Model initialization complete")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_grid(self) -> None:
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

    def _initialize_parameters(self) -> None:
        """Load and process model parameters"""
        self.logger.info("Loading model parameters")

        # Load parameter files
        for name, path in self.config.input.lookup_tables.items():
            self.parameters.add_file(str(path))

        # Process parameters
        self.parameters.munge_files()

        # Create parameter dictionaries
        landuse_params = self._process_landuse_parameters()
        soil_params = self._process_soil_parameters()

        # Initialize domain with parameters
        self.domain.initialize_parameters(landuse_params, soil_params)

    def _load_input_data(self) -> None:
        """Load required input data"""
        self.logger.info("Loading input data")

        try:
            # Load required grids
            self.landuse_indices = self._load_grid(
                self.config.input.landuse_grid,
                GridDataType.INTEGER
            )
            self.soil_indices = self._load_grid(
                self.config.input.soils_grid,
                GridDataType.INTEGER
            )

            # Load optional grids with defaults
            self.elevation = self._load_grid(
                self.config.input.elevation_grid,
                GridDataType.REAL,
                default_value=0.0
            )
            self.latitude = self._load_grid(
                self.config.input.latitude_grid,
                GridDataType.REAL,
                default_value=20.7  # Default to Maui latitude
            )

        except Exception as e:
            self.logger.error(f"Failed to load input data: {str(e)}")
            raise

    def _initialize_domain(self) -> None:
        """Initialize model domain components"""
        self.logger.info("Initializing model domain")

        # Initialize simulation period
        self.domain.initialize_simulation_period(
            self.config.start_date,
            self.config.end_date
        )

        # Initialize modules with data
        self.domain.initialize_modules(
            landuse_indices=self.landuse_indices,
            soil_indices=self.soil_indices,
            elevation=self.elevation,
            latitude=self.latitude,
            fragments_file=self.config.input.fragments_file
        )

        # Initialize optional features
        self.domain.initialize_crop_coefficients(
            use_crop_coefficients=self.config.use_crop_coefficients,
            dynamic_rooting=self.config.dynamic_rooting
        )

        if self.config.routing_enabled:
            self.domain.initialize_routing(
                self.flow_direction,
                self.routing_fraction
            )

    def _initialize_output(self) -> None:
        """Initialize model output configuration"""
        self.logger.info("Initializing output configuration")

        self.domain.initialize_output(
            directory=self.config.output.directory,
            prefix=self.config.output.prefix,
            variables=self.config.output.variables,
            compression=self.config.output.compression
        )

        # Create output directory if needed
        self.config.output.directory.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run model simulation"""
        if not self.initialized:
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

    def _process_timestep(self, date: datetime) -> None:
        """Process single timestep"""
        # Update model date
        self.domain.update_date(date)

        # Get weather data first
        self.domain.get_weather_data(date)

        # Run daily calculations
        self.daily_calc.perform_daily_calculation(date)

        # Write output if configured
        if self.config.output.write_daily:
            self._write_output(date)

    def _write_output(self, date: datetime) -> None:
        """Write output for current timestep"""
        try:
            output_path = (self.config.output.directory /
                           f"{self.config.output.prefix}_{date:%Y%m%d}.nc")

            # Prepare output data
            output_data = {}
            for var in self.config.output.variables:
                if hasattr(self.domain, var.lower()):
                    output_data[var] = getattr(self.domain, var.lower())

            # Write using data manager
            self.data_manager.write(output_data, output_path)

        except Exception as e:
            self.logger.error(f"Failed to write output for {date}: {str(e)}")
            raise

    def _load_grid(self, path: Optional[Path], dtype: GridDataType,
                   default_value: Optional[float] = None) -> np.ndarray:
        """Load grid data with validation"""
        if path is None:
            if default_value is None:
                raise ValueError("Path or default value must be provided")
            return np.full((self.domain.number_of_rows,
                            self.domain.number_of_columns), default_value)

        data = self.data_manager.load(path)
        if data.shape != (self.domain.number_of_rows,
                          self.domain.number_of_columns):
            raise ValueError(f"Grid shape mismatch: {data.shape} vs "
                             f"({self.domain.number_of_rows}, "
                             f"{self.domain.number_of_columns})")
        return data

    def _process_landuse_parameters(self) -> Dict[int, Dict]:
        """Process landuse parameters from parameter files"""
        landuse_params = {}
        landuse_table = self.parameters.get_parameters('landuse')

        for row in landuse_table:
            landuse_id = int(row['LU_code'])
            landuse_params[landuse_id] = {
                'runoff': {
                    'curve_number': float(row['curve_number']),
                    'initial_abstraction_ratio': float(row.get('ia_ratio', 0.2)),
                    'depression_storage': float(row.get('depression_storage', 0.1))
                },
                'crop': {
                    'crop_coefficient': float(row.get('crop_coefficient', 1.0)),
                    'root_depth': float(row.get('root_depth', 0.0)),
                    'growing_season_start': row.get('growing_season_start'),
                    'growing_season_end': row.get('growing_season_end')
                },
                'interception': {
                    'capacity': float(row.get('interception_capacity', 0.0)),
                    'coverage': float(row.get('vegetation_coverage', 1.0))
                }
            }

        return landuse_params

    def _process_soil_parameters(self) -> Dict[int, Dict]:
        """Process soil parameters from parameter files"""
        soil_params = {}
        soil_table = self.parameters.get_parameters('soils')

        for row in soil_table:
            soil_id = int(row['soil_id'])
            soil_params[soil_id] = {
                'infiltration': {
                    'hydraulic_conductivity': float(row['k_sat']),
                    'wetting_front_suction': float(row['suction']),
                    'initial_moisture': float(row.get('initial_moisture', 0.0))
                },
                'soil': {
                    'porosity': float(row['porosity']),
                    'field_capacity': float(row['field_capacity']),
                    'wilting_point': float(row['wilting_point']),
                    'residual_moisture': float(row.get('residual_moisture', 0.0))
                }
            }

        return soil_params

    def cleanup(self) -> None:
        """Clean up model resources"""
        self.logger.info("Cleaning up model resources")

        try:
            # Clean up domain
            self.domain.cleanup()

            # Close any open files
            if self._nc_out is not None:
                self._nc_out.close()

            # Clear data manager cache
            self.data_manager.clear_cache()

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise


def run_swb_model(config_path: Path, log_dir: Optional[Path] = None) -> None:
    """Convenience function to run SWB model"""
    model = SWBModel(log_dir)
    try:
        model.initialize(config_path)
        model.run()
    except Exception as e:
        logging.error(f"Model run failed: {str(e)}")
        raise