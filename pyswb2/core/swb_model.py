from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
import numpy as np
import netCDF4 as nc

from .configuration import ConfigurationManager, ModelConfig
from .model_domain import ModelDomain
from .grid import Grid, GridDataType
from .daily_calculation import DailyCalculation
from .data_catalog import DataCatalog, DataCatalogEntry
from .parameters import Parameters
from .logging_config import ParameterLogger

@dataclass
class GridData:
    data: np.ndarray
    metadata: Dict[str, Union[str, float, int]]
    transform: Optional[Dict[str, float]] = None

class DataInterface:
    def read(self, path: Path) -> GridData:
        raise NotImplementedError
        
    def write(self, data: GridData, path: Path) -> None:
        raise NotImplementedError

class ASCIIGridInterface(DataInterface):
    def read(self, path: Path) -> GridData:
        header = {}
        with open(path) as f:
            for _ in range(6):
                key, val = f.readline().split()
                header[key.lower()] = float(val)
        data = np.loadtxt(f)
        return GridData(
            data=data,
            metadata={
                'nx': int(header['ncols']),
                'ny': int(header['nrows']),
                'x_ll': header['xllcorner'],
                'y_ll': header['yllcorner'],
                'cell_size': header['cellsize'],
                'nodata': header['nodata_value']
            }
        )

class NetCDFInterface(DataInterface):
    def write(self, data: GridData, path: Path) -> None:
        with nc.Dataset(path, 'w') as ds:
            ds.createDimension('y', data.metadata['ny'])
            ds.createDimension('x', data.metadata['nx'])
            var = ds.createVariable('data', 'f4', ('y', 'x'))
            var[:] = data.data
            for key, value in data.metadata.items():
                setattr(var, key, value)

class DataManager:
    def __init__(self):
        self.interfaces: Dict[str, DataInterface] = {
            '.asc': ASCIIGridInterface(),
            '.nc': NetCDFInterface()
        }
        self.cache: Dict[str, GridData] = {}
        
    def load(self, path: Path, transform: Optional[Dict[str, float]] = None) -> GridData:
        if str(path) in self.cache:
            return self.cache[str(path)]
            
        interface = self.interfaces.get(path.suffix.lower())
        if not interface:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        data = interface.read(path)
        if transform:
            data.transform = transform
            data.data = self._apply_transform(data.data, transform)
            
        self.cache[str(path)] = data
        return data
        
    def save(self, data: GridData, path: Path) -> None:
        interface = self.interfaces.get(path.suffix.lower())
        if not interface:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        interface.write(data, path)
        
    def _apply_transform(self, data: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
        result = data.copy()
        if 'scale' in transform:
            result *= transform['scale']
        if 'offset' in transform:
            result += transform['offset']
        if 'missing_value' in transform:
            result[result == transform['missing_value']] = np.nan
        return result

class ModelDataFlow:
    def __init__(self):
        self.data_manager = DataManager()
        self.validation_rules: Dict[str, List[callable]] = {}
        
    def register_validation(self, data_type: str, rules: List[callable]) -> None:
        self.validation_rules[data_type] = rules
        
    def load_and_validate(self, path: Path, data_type: str) -> GridData:
        data = self.data_manager.load(path)
        if data_type in self.validation_rules:
            for rule in self.validation_rules[data_type]:
                if not rule(data):
                    raise ValueError(f"Validation failed for {path}")
        return data
        
    def prepare_output(self, data: Dict[str, np.ndarray], metadata: Dict[str, Dict]) -> Dict[str, GridData]:
        return {
            key: GridData(array, metadata[key])
            for key, array in data.items()
            if key in metadata
        }

class SWBModel:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.domain = ModelDomain()
        self.grid = Grid()
        self.daily_calc = DailyCalculation(self.domain)
        self.data_catalog = DataCatalog()
        self.parameters = Parameters()
        self.data_flow = ModelDataFlow()
        self.logger = ParameterLogger()
        self.precipitation = Precipitation(grid_shape=self.grid.grid_shape)
        
    def initialize(self, config_path: Path) -> None:
        # Load and validate configuration
        self.config = self.config_manager.load(config_path)
        
        # Set up validation rules
        self._setup_validations()
        
        # Initialize grid
        self.grid.initialize(
            nx=self.config.grid.nx,
            ny=self.config.grid.ny,
            x0=self.config.grid.x0,
            y0=self.config.grid.y0,
            cell_size=self.config.grid.cell_size,
            proj4_string=self.config.grid.proj4_string
        )
        
        # Initialize domain
        self.domain.initialize_grid(
            nx=self.grid.nx,
            ny=self.grid.ny,
            x_ll=self.grid.x0,
            y_ll=self.grid.y0,
            cell_size=self.grid.cell_size
        )
        
        # Load input data
        self._load_input_data()
        
        # Initialize methods and output
        self.domain.initialize_methods()
        self._initialize_output()
        
    def _setup_validations(self):
        """Set up data validation rules"""
        self.data_flow.register_validation('landuse', [
            lambda x: x.data.dtype == np.int32,
            lambda x: np.all(x.data >= 0)
        ])
        
        self.data_flow.register_validation('soils', [
            lambda x: x.data.dtype == np.int32,
            lambda x: np.all(x.data > 0)
        ])
        
        self.data_flow.register_validation('awc', [
            lambda x: x.data.dtype in [np.float32, np.float64],
            lambda x: np.all(~np.isnan(x.data))
        ])
        
    def _load_input_data(self) -> None:
        # Load required input grids
        self.landuse = self.data_flow.load_and_validate(
            self.config.input.landuse_grid, 'landuse')
        self.soils = self.data_flow.load_and_validate(
            self.config.input.soils_grid, 'soils')
            
        # Load optional AWC data
        if self.config.input.awc_grid:
            self.awc = self.data_flow.load_and_validate(
                self.config.input.awc_grid, 'awc')
            
        # Load lookup tables
        for name, path in self.config.input.lookup_tables.items():
            self.parameters.add_file(str(path))
            
    def _initialize_output(self) -> None:
        self.domain.initialize_output(
            self.config.output.directory,
            self.config.output.prefix,
            self.config.output.variables,
            self.config.output.compression
        )
        
    def run(self) -> None:
        """Run model simulation"""
        current_date = self.config.start_date
        
        while current_date <= self.config.end_date:
            self.logger.info(f"Processing: {current_date.strftime('%Y-%m-%d')}")
            
            # Get weather data
            self.domain.get_weather_data(current_date)
            
            # Run daily calculations
            self.daily_calc.perform_daily_calculation(current_date, self.config)
            
            # Prepare and write output
            output_data = self._prepare_timestep_output()
            self._write_output(current_date, output_data)
            
            # Advance time
            current_date += timedelta(days=1)
            
    def _prepare_timestep_output(self) -> Dict[str, np.ndarray]:
        """Prepare output data for current timestep"""
        return {
            var: getattr(self.domain, var.lower())
            for var in self.config.output.variables
            if hasattr(self.domain, var.lower())
        }
        
    def _write_output(self, date: datetime, timestep_data: Dict[str, np.ndarray]) -> None:
        """Write output for current timestep"""
        output_data = self.data_flow.prepare_output(
            timestep_data,
            self.domain.output_metadata
        )
        
        output_path = self.config.output.directory / f"{self.config.output.prefix}_{date:%Y%m%d}.nc"
        self.data_flow.data_manager.save(output_data, output_path)
        
    def cleanup(self) -> None:
        """Cleanup model resources"""
        self.logger.info("Cleaning up model resources")
        if hasattr(self.domain, '_nc_out'):
            self.domain._nc_out.close()
        self.data_flow.data_manager.cache.clear()
