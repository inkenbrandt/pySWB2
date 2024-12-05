from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import netCDF4 as nc
import os

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
        
        # Masks and indices
        self.active: NDArray[np.bool_] = None
        self.landuse_code: NDArray[np.int32] = None
        self.landuse_index: NDArray[np.int32] = None
        self.soil_group: NDArray[np.int32] = None
        
        # State variables 
        self.soil_storage: NDArray[np.float32] = None
        self.soil_storage_max: NDArray[np.float32] = None
        self.surface_storage: NDArray[np.float32] = None
        self.snow_storage: NDArray[np.float32] = None
        self.interception_storage: NDArray[np.float32] = None
        
        # Water fluxes
        self.gross_precip: NDArray[np.float32] = None
        self.rainfall: NDArray[np.float32] = None
        self.snowfall: NDArray[np.float32] = None
        self.runoff: NDArray[np.float32] = None
        self.runon: NDArray[np.float32] = None
        self.infiltration: NDArray[np.float32] = None
        self.net_infiltration: NDArray[np.float32] = None
        
        # Energy and ET
        self.reference_et0: NDArray[np.float32] = None
        self.actual_et: NDArray[np.float32] = None
        self.tmin: NDArray[np.float32] = None
        self.tmax: NDArray[np.float32] = None
        self.tmean: NDArray[np.float32] = None
        
        # Output specifications
        self.outspecs: Dict[OutputType, OutputSpec] = {}
        self._initialize_output_specs()
        
    def _initialize_output_specs(self):
        """Initialize output specifications"""
        specs = [
            OutputSpec("gross_precipitation", "inches", 0.0, 2000.0, True),
            OutputSpec("rainfall", "inches", 0.0, 2000.0, True),
            OutputSpec("snowfall", "inches", 0.0, 2000.0, True),
            # ... Add other specs
        ]
        for type_enum, spec in zip(OutputType, specs):
            self.outspecs[type_enum] = spec
            
    def initialize_grid(self, nx: int, ny: int, x_ll: float, y_ll: float, 
                       cell_size: float, proj4: str = ""):
        """Initialize model grid properties"""
        self.number_of_columns = nx
        self.number_of_rows = ny
        self.gridcellsize = cell_size
        self.x_ll = x_ll
        self.y_ll = y_ll
        self.proj4_string = proj4
        
        # Initialize masks
        self.active = np.ones((ny, nx), dtype=bool)
        
    def initialize_arrays(self):
        """Initialize model state arrays"""
        n_active = np.count_nonzero(self.active)
        
        # Initialize all arrays to proper sizes
        self.landuse_code = np.zeros(n_active, dtype=np.int32)
        self.landuse_index = np.zeros(n_active, dtype=np.int32)
        self.soil_group = np.zeros(n_active, dtype=np.int32)
        
        # Water storages
        self.soil_storage = np.zeros(n_active, dtype=np.float32)
        self.soil_storage_max = np.zeros(n_active, dtype=np.float32)
        self.surface_storage = np.zeros(n_active, dtype=np.float32)
        self.snow_storage = np.zeros(n_active, dtype=np.float32)
        
        # Water fluxes 
        self.gross_precip = np.zeros(n_active, dtype=np.float32)
        self.rainfall = np.zeros(n_active, dtype=np.float32)
        self.snowfall = np.zeros(n_active, dtype=np.float32)
        self.runoff = np.zeros(n_active, dtype=np.float32)
        self.runon = np.zeros(n_active, dtype=np.float32)
        
    def set_inactive_cells(self, landuse_grid: NDArray, 
                          soil_grid: NDArray,
                          awc_grid: Optional[NDArray] = None):
        """Set inactive grid cells based on input data"""
        self.active &= (landuse_grid >= 0)
        self.active &= (soil_grid > 0)
        if awc_grid is not None:
            self.active &= (awc_grid >= 0)
            
    def get_weather_data(self, date: datetime):
        """Update weather data for given date"""
        self._get_precipitation_data(date)
        self._get_temperature_data(date)
        self._calculate_mean_temperature()
        
    def write_output(self, date: datetime, output_dir: str):
        """Write output for current timestep"""
        if not hasattr(self, '_nc_out'):
            self._initialize_netcdf(output_dir)
            
        # Add current timestep data
        time_index = (date - self._start_date).days
        
        # Write active output variables
        for out_type, spec in self.outspecs.items():
            if spec.is_active:
                var_name = spec.variable_name
                if hasattr(self, var_name.lower()):
                    data = getattr(self, var_name.lower())
                    self._nc_out.variables[var_name][time_index,:,:] = self._pack_to_grid(data)
                    
    def _pack_to_grid(self, data: NDArray) -> NDArray:
        """Pack 1D array back to 2D grid"""
        grid = np.full((self.number_of_rows, self.number_of_columns), 
                      self._fill_value, dtype=np.float32)
        grid[self.active] = data
        return grid
        
    def _initialize_netcdf(self, output_dir: str):
        """Initialize NetCDF output file"""
        os.makedirs(output_dir, exist_ok=True)
        nc_file = os.path.join(output_dir, 'output.nc')
        
        self._nc_out = nc.Dataset(nc_file, 'w')
        self._nc_out.createDimension('time', None)
        self._nc_out.createDimension('y', self.number_of_rows)
        self._nc_out.createDimension('x', self.number_of_columns)
        
        # Create variables for all active outputs
        for out_type, spec in self.outspecs.items():
            if spec.is_active:
                var = self._nc_out.createVariable(spec.variable_name, 
                                                'f4', ('time','y','x'))
                var.units = spec.variable_units
                var.valid_min = spec.valid_minimum
                var.valid_max = spec.valid_maximum

    def initialize_methods(self):
        """Initialize all model calculation methods"""
        methods = {
            'interception': self._init_interception_bucket,
            'runoff': self._init_runoff_curve_number,
            'reference_et': self._init_et_hargreaves,
            'snowfall': self._init_snowfall_original,
            'snowmelt': self._init_snowmelt_original
        }
        for name, method in methods.items():
            method()

    def calculate_daily(self):
        """Run daily water balance calculations"""
        # Order matches Fortran implementation
        self._calc_interception()
        self._calc_snowfall()
        self._calc_snowmelt()
        self._calc_reference_et()
        self._calc_actual_et()
        self._calc_runoff()
        self._update_storages()

    def routing_methods(self):
        """Flow routing between cells"""
        self._init_routing_d8()
        self._calc_routing_d8()
        self._update_routing_fractions()

    def process_weather(self):
        """Process raw weather inputs"""
        self._adjust_temperatures()
        self._partition_precipitation()
        self._calculate_potential_et()