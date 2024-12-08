from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any
import numpy as np
import logging
import pandas as pd

from .configuration import ConfigurationManager, ModelConfig
from .model_domain import ModelDomain
from .daily_calculation import DailyCalculation
from .runoff import RunoffParameters 
from .agriculture import CropParameters
from .interception import GashParameters
from .infiltration import InfiltrationParameters
from .soil import SoilParameters
from .logging_config import ParameterLogger

class SWBRunner:
    """Main class for running SWB model simulations with Maui configuration"""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize SWB runner
        
        Args:
            log_dir: Optional directory for log files
        """

        self.config_manager = ConfigurationManager()
        self.model = ModelDomain(log_dir)  # Pass log_dir here
        self.logger = ParameterLogger(log_dir)
        
    def parse_maui_control_file(self, control_file: Path) -> Dict[str, Any]:
        """Parse Maui control file format
        
        Args:
            control_file: Path to control file
            
        Returns:
            Dictionary of configuration parameters
        """
        config = {}
        current_section = None
        
        with open(control_file) as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith(('#', '!', '$', '%', '*', '-', '(', '[')):
                    continue
                    
                # Split into key-value pairs
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    config[key.lower()] = value
                    
        return config

    def parse_maui_lookup_table(self, lookup_file: Path) -> Dict[int, Dict[str, Any]]:
        """Parse Maui landuse lookup table

        Args:
            lookup_file: Path to lookup table

        Returns:
            Dictionary mapping landuse codes to parameter dictionaries
        """
        # Read lookup table
        df = pd.read_csv(lookup_file, sep='\t', comment='#')

        # Convert to dictionary of parameter dictionaries
        landuse_params = {}

        for _, row in df.iterrows():
            lu_code = int(row['LU code'])

            # Process curve numbers - handle space in column names
            curve_numbers = []
            for i in range(1, 5):
                cn_col = f'CN {i}'  # Space instead of underscore
                if cn_col in row:
                    curve_numbers.append(float(row[cn_col]))

            # Process max infiltration rates
            max_infil = []
            for i in range(1, 5):
                infil_col = f'Max net infil {i}'  # Space between words
                if infil_col in row:
                    max_infil.append(float(row[infil_col]))

            # Create parameter dictionary
            landuse_params[lu_code] = {
                'runoff': {
                    'curve_number': curve_numbers[0] if curve_numbers else 70.0,  # Default CN if not found
                    'initial_abstraction_ratio': 0.2,
                    'depression_storage': float(row['Surface Storage Max']),
                    'impervious_fraction': float(row['Assumed Imperviousness'])
                    if not pd.isna(row['Assumed Imperviousness']) and row['Assumed Imperviousness'] != '**'
                    else 0.0
                },
                'interception': {
                    'canopy_storage_capacity': float(row['Canopy_Storage_Capacity']),
                    'trunk_storage_capacity': float(row['Trunk_Storage_Capacity']),
                    'stemflow_fraction': float(row['Stemflow_Fraction']),
                    'interception_storage_max_growing': float(row['Interception_Growing']),
                    'interception_storage_max_nongrowing': float(row['Interception_Nongrowing'])
                },
                'crop': {
                    'gdd_base': float(row['GDD_Base_Temperature']),
                    'gdd_max': float(row['GDD_Maximum_Temperature']),
                    'growing_season_start': row['First_day_of_growing_season'],
                    'growing_season_end': row['Last_day_of_growing_season'],
                    'initial_root_depth': float(row['Rooting_depth_inches']) if not pd.isna(
                        row['Rooting_depth_inches']) else 0.0,
                    'max_root_depth': float(row['Rooting_depth_inches']) if not pd.isna(
                        row['Rooting_depth_inches']) else 0.0,
                    'crop_coefficient': float(row['Kcb_mid']) if not pd.isna(row['Kcb_mid']) else 1.0
                }
            }

            # Add max infiltration rates if available
            if max_infil:
                landuse_params[lu_code]['infiltration'] = {
                    'maximum_rates': max_infil
                }

            # Process monthly crop coefficients if available
            monthly_kcb = {}
            for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                col = f'Kcb_{month}'
                if col in row and not pd.isna(row[col]):
                    monthly_kcb[month] = float(row[col])

            if monthly_kcb:
                landuse_params[lu_code]['crop']['monthly_coefficients'] = monthly_kcb

            # Process irrigation parameters if available
            irrigation_params = {}
            if 'Max_allowable_depletion' in row:
                irrigation_params['max_allowable_depletion'] = float(row['Max_allowable_depletion'])
            if 'Irrigation_efficiency' in row:
                irrigation_params['efficiency'] = float(row['Irrigation_efficiency'])
            if 'Application Amount' in row and not pd.isna(row['Application Amount']):
                irrigation_params['application_amount'] = float(row['Application Amount'])
            if 'Monthly Application Schedule' in row and not pd.isna(row['Monthly Application Schedule']):
                irrigation_params['monthly_schedule'] = str(row['Monthly Application Schedule'])

            if irrigation_params:
                landuse_params[lu_code]['irrigation'] = irrigation_params

        return landuse_params

    def configure(self, control_file: Path) -> None:
        """Configure model from Maui control file
        
        Args:
            control_file: Path to control file
        """
        self.logger.info(f"Loading configuration from {control_file}")
        
        # Parse Maui control file
        config = self.parse_maui_control_file(control_file)
        
        # Parse grid configuration
        grid_parts = config['grid'].split()
        nx, ny = int(grid_parts[0]), int(grid_parts[1])
        x0, y0 = float(grid_parts[2]), float(grid_parts[3])
        cell_size = float(grid_parts[4])
        
        # Initialize model grid
        self.logger.info("Initializing model grid")
        self.model.initialize_grid(
            nx=nx,
            ny=ny,
            x_ll=x0, 
            y_ll=y0,
            cell_size=cell_size,
            proj4=config.get('base_projection_definition', '')
        )
        
        # Parse dates
        start_date = datetime.strptime(config['start_date'], '%m/%d/%Y')
        end_date = datetime.strptime(config['end_date'], '%m/%d/%Y')
        
        # Initialize simulation period
        self.model.initialize_simulation_period(start_date, end_date)
        
        # Load landuse lookup table
        lookup_file = Path(config['land_use_lookup_table'])
        self.landuse_params = self.parse_maui_lookup_table(lookup_file)
        
        # Load and process input grids
        self._load_input_grids(config)
        
        # Initialize model modules
        self._initialize_modules(config)
        
    def _load_input_grids(self, config: Dict[str, Any]) -> None:
        """Load required input grids"""
        self.logger.info("Loading input grids")
        
        # Load landuse grid
        landuse_path = config['land_use'].split()[1]
        landuse = np.loadtxt(landuse_path, skiprows=6)
        self.landuse_indices = landuse.flatten()
        
        # Load soils grid
        soils_path = config['hydrologic_soils_group'].split()[1]
        soils = np.loadtxt(soils_path, skiprows=6)
        self.soil_indices = soils.flatten()
        
        # Load soil storage max if specified
        if 'soil_storage_max' in config:
            storage_path = config['soil_storage_max'].split()[1]
            self.soil_storage_max = np.loadtxt(storage_path, skiprows=6).flatten()
        else:
            self.soil_storage_max = None

    def _initialize_modules(self, config: Dict[str, Any]) -> None:
        """Initialize all model modules"""
        self.logger.info("Initializing model modules")

        # Convert parameters to module-specific classes
        landuse_module_params = {}

        for code, params in self.landuse_params.items():
            # Handle crop parameters
            crop_params = params['crop'].copy()
            if 'monthly_coefficients' in crop_params:
                crop_params.pop('monthly_coefficients')

            landuse_module_params[code] = {
                'runoff': RunoffParameters(**params['runoff']),
                'interception': GashParameters(**params['interception']),
                'crop': CropParameters(**crop_params)
            }

            # Add optional parameters
            if 'irrigation' in params:
                landuse_module_params[code]['irrigation'] = params['irrigation']

            if 'infiltration' in params:
                landuse_module_params[code]['infiltration'] = params['infiltration']

        # Initialize model parameters
        self.model.initialize_parameters(landuse_module_params, {})

        # Initialize modules with configuration
        self.model.initialize_modules(config)

        # Initialize model data
        self.model.initialize_modules_data(
            landuse_indices=self.landuse_indices,
            soil_indices=self.soil_indices,
            elevation=np.zeros_like(self.landuse_indices),  # Default elevation
            latitude=np.full_like(self.landuse_indices, 20.7),  # Maui latitude
            fragments_file=Path(config['fragments_daily_file'].strip())
            if 'fragments_daily_file' in config else None
        )

        # Configure crop methods
        self.model.initialize_crop_coefficients(
            use_crop_coefficients=config.get('crop_coefficient_method', 'NONE') != 'NONE',
            dynamic_rooting=True
        )

    def run(self) -> None:
        """Run model simulation"""
        daily_calc = DailyCalculation(self.model)
        
        self.logger.info(f"Starting simulation from {self.model.start_date} to {self.model.end_date}")
        
        current_date = self.model.start_date
        while current_date <= self.model.end_date:
            self.logger.debug(f"Processing {current_date}")
            
            # Get weather data
            self.model.get_weather_data(current_date)
            
            # Update model date
            self.model.update_date(current_date)
            
            # Run daily calculations
            daily_calc.perform_daily_calculation(current_date)
            
            # Advance to next day
            current_date += timedelta(days=1)
            
        self.logger.info("Simulation complete")
        
    def cleanup(self) -> None:
        """Clean up model resources"""
        self.logger.info("Cleaning up")
        self.model.cleanup()
        
def run_maui_swb(control_file: Union[str, Path], log_dir: Optional[Path] = None) -> None:
    """Convenience function to run Maui SWB model
    
    Args:
        control_file: Path to Maui control file
        log_dir: Optional directory for log files
    """
    # Initialize runner
    runner = SWBRunner(log_dir)
    
    try:
        # Configure and run model
        runner.configure(Path(control_file))
        runner.run()
    finally:
        runner.cleanup()
