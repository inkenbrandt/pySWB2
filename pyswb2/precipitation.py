import numpy as np
from pathlib import Path
import math

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Union, Tuple

import netCDF4 as nc
from numpy.typing import NDArray


@dataclass
class PrecipitationReader:
    """Reader for precipitation data in various formats"""

    def __init__(self, file_path: Path):
        """Initialize reader with file path

        Args:
            file_path: Path to precipitation data file
        """
        self.file_path = file_path
        self.file_type = self._detect_file_type()
        self.nc_dataset = None
        self._initialize_reader()

    def _detect_file_type(self) -> str:
        """Detect file type from extension"""
        suffix = self.file_path.suffix.lower()
        if suffix in ['.nc', '.nc4', '.netcdf']:
            return 'netcdf'
        elif suffix in ['.asc', '.txt']:
            return 'ascii'
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _initialize_reader(self) -> None:
        """Initialize appropriate reader based on file type"""
        if self.file_type == 'netcdf':
            self.nc_dataset = nc.Dataset(self.file_path, 'r')

    def read_timestep(self, date: datetime) -> NDArray[np.float32]:
        """Read precipitation data for a specific timestep

        Args:
            date: Date to read data for

        Returns:
            Array of precipitation values
        """
        if self.file_type == 'netcdf':
            return self._read_netcdf_timestep(date)
        else:
            return self._read_ascii_timestep(date)

    def _read_netcdf_timestep(self, date: datetime) -> NDArray[np.float32]:
        """Read timestep from NetCDF file"""
        if self.nc_dataset is None:
            raise RuntimeError("NetCDF dataset not initialized")

        # Find time variable
        time_var = self.nc_dataset.variables.get('time')
        if time_var is None:
            raise ValueError("NetCDF file missing time variable")

        # Convert date to time index
        time_units = time_var.units
        calendar = getattr(time_var, 'calendar', 'standard')

        # Find time index for requested date
        target_time = nc.date2num(date, time_units, calendar)
        time_index = np.abs(time_var[:] - target_time).argmin()

        # Get precipitation variable (try common names)
        precip_names = ['precip', 'precipitation', 'PRECIP', 'PRCP', 'pr']
        precip_var = None
        for name in precip_names:
            if name in self.nc_dataset.variables:
                precip_var = self.nc_dataset.variables[name]
                break

        if precip_var is None:
            raise ValueError("Could not find precipitation variable in NetCDF file")

        # Read data for timestep
        data = precip_var[time_index, :, :]

        # Handle fill values/missing data
        fill_value = getattr(precip_var, '_FillValue', None)
        if fill_value is not None:
            data = np.where(data == fill_value, 0.0, data)

        return data.astype(np.float32)

    def _read_ascii_timestep(self, date: datetime) -> NDArray[np.float32]:
        """Read timestep from ASCII file"""
        # Construct filename with date
        date_str = date.strftime('%Y%m%d')
        file_pattern = self.file_path.stem + date_str + self.file_path.suffix
        timestep_file = self.file_path.parent / file_pattern

        if not timestep_file.exists():
            raise FileNotFoundError(f"Precipitation file not found: {timestep_file}")

        # Read ASCII grid
        header = {}
        with open(timestep_file) as f:
            # Read header
            for _ in range(6):
                key, val = f.readline().split()
                header[key.lower()] = float(val)

            # Read data
            data = np.loadtxt(f, dtype=np.float32)

            # Replace nodata values with 0
            nodata = header['nodata_value']
            data = np.where(data == nodata, 0.0, data)

        return data

    def close(self) -> None:
        """Close open file handles"""
        if self.nc_dataset is not None:
            self.nc_dataset.close()
            self.nc_dataset = None

    def __del__(self):
        """Ensure resources are cleaned up"""
        self.close()

@dataclass
class FragmentSet:
    rain_gage_zone: int
    start_record: Dict[int, int]
    number_of_fragments: Dict[int, int]

@dataclass
class PrecipitationData:
    """Container for precipitation data using arrays"""
    date: datetime
    gross_precip: NDArray[np.float32]
    rainfall: NDArray[np.float32]
    snowfall: NDArray[np.float32]
    net_rainfall: NDArray[np.float32]
    net_snowfall: NDArray[np.float32]

class Precipitation:
    def __init__(self, grid_shape: tuple):
        # Previous initialization code remains the same...
        self.FREEZING_F = 32.0
        self.NEAR_ZERO = 1.0e-9

        # Grid initialization
        self.grid_shape = grid_shape
        self.domain_size = grid_shape[0] * grid_shape[1]  # Total number of cells

        # Initialize arrays with flattened shape
        self.precipitation = np.zeros(grid_shape, dtype=np.float32)
        self.snowfall = np.zeros(grid_shape, dtype=np.float32)
        self.rainfall = np.zeros(grid_shape, dtype=np.float32)

        # Method of Fragments variables
        self.fragments = []
        self.fragments_sets = {}
        self.current_fragments = None
        self.rainfall_zones = None
        self.use_fragments = False

        # Initialize random_values with a default shape to match max possible zones
        # This will be resized in initialize() when actual zones are known
        self.random_values = np.random.random(100)  # Default size, will be resized

        # State tracking
        self._current_data = None
        self.simulation_number = 1

        # Precipitation reader
        self.precip_reader = None

    def _ensure_shape(self, array: NDArray) -> NDArray:
        """Ensure array is properly shaped and flattened

        Args:
            array: Input array to reshape

        Returns:
            Reshaped array matching domain size
        """
        if array is None:
            return np.zeros(self.domain_size, dtype=np.float32)

        # If array is already 1D and correct size, return as-is
        if array.shape == (self.domain_size,):
            return array

        # If array matches grid shape, flatten it
        if array.shape == self.grid_shape:
            return array.ravel()

        # If array is different shape, try to reshape
        try:
            reshaped = array.reshape(self.grid_shape)
            return reshaped.ravel()
        except ValueError:
            raise ValueError(
                f"Cannot reshape array of shape {array.shape} to match domain "
                f"size {self.domain_size} (grid shape {self.grid_shape})"
            )

    def initialize(self, precip_file: Path, rainfall_zones: np.ndarray,
                   random_start: int = 12345, use_fragments: bool = True) -> None:
        """Initialize precipitation module with data source

        Args:
            precip_file: Path to precipitation data file (ASCII or NetCDF)
            rainfall_zones: Array of rainfall zone indices
            random_start: Random seed for fragments
            use_fragments: Whether to use method of fragments
        """
        self.rainfall_zones = rainfall_zones
        self.use_fragments = use_fragments

        # Get number of unique zones and resize random values array
        max_zone = int(np.max(rainfall_zones))
        self.random_values = np.random.random(max_zone)

        # Set random seed
        np.random.seed(random_start)

        if use_fragments:
            # Initialize fragments arrays if using method of fragments
            self._read_daily_fragments(precip_file)
            self._process_fragment_sets()
            self.current_fragments = [None] * max_zone
        else:
            # Initialize precipitation reader for direct data
            self.precip_reader = PrecipitationReader(precip_file)


    def _read_daily_fragments(self, filename: Path) -> None:
        """Read daily precipitation fragments from file"""
        if not filename.exists():
            raise FileNotFoundError(f"Fragments file not found: {filename}")

        with open(filename) as f:
            for line in f:
                if line.startswith(("#", "%", "!")):
                    continue

                parts = line.split()
                month = int(parts[0])
                rain_gage_zone = int(parts[1])
                fragment_set = int(parts[2])
                fragment_values = [float(x) for x in parts[3:]]

                # Validate and normalize fragment values
                fragment_values = [max(0.0, min(1.0, x)) for x in fragment_values]
                if month == 2:
                    self._normalize_february_fragments(fragment_values)

                self.fragments.append({
                    'month': month,
                    'zone': rain_gage_zone,
                    'set': fragment_set,
                    'values': fragment_values
                })

    def _process_fragment_sets(self) -> None:
        """Process fragments into organized sets by zone and month"""
        max_zone = max(f['zone'] for f in self.fragments)
        for zone in range(1, max_zone + 1):
            fragment_set = FragmentSet(
                rain_gage_zone=zone,
                start_record={m: 0 for m in range(1, 13)},
                number_of_fragments={m: 0 for m in range(1, 13)}
            )
            
            # Find fragments for each month in this zone
            for month in range(1, 13):
                fragments = [i for i, f in enumerate(self.fragments) 
                           if f['zone'] == zone and f['month'] == month]
                if fragments:
                    fragment_set.start_record[month] = min(fragments)
                    fragment_set.number_of_fragments[month] = len(fragments)
                    
            self.fragments_sets[zone] = fragment_set

    def calculate_snowmelt(self, tmin: float, tmax: float) -> float:
        """Calculate potential snowmelt"""
        if tmax > self.FREEZING_F:
            melt_temp = max(0.0, (tmin + tmax) / 2 - self.FREEZING_F)
            return 1.5 * melt_temp
        return 0.0

    def _partition_precipitation(self, tmin: NDArray[np.float32],
                                 tmax: NDArray[np.float32],
                                 interception: NDArray[np.float32],
                                 gross_precip: NDArray[np.float32]) -> Tuple[NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32]]:
        """Partition precipitation into rain and snow based on temperature

        Args:
            tmin: Array of minimum temperatures
            tmax: Array of maximum temperatures
            interception: Array of interception values
            gross_precip: Array of gross precipitation values

        Returns:
            Tuple of (snowfall, net_snowfall, rainfall, net_rainfall) arrays
        """
        # Initialize output arrays
        snowfall = np.zeros_like(gross_precip)
        rainfall = np.zeros_like(gross_precip)

        # All snow when max temp is below freezing
        snow_mask = tmax <= self.FREEZING_F
        snowfall[snow_mask] = gross_precip[snow_mask]

        # All rain when min temp is above freezing
        rain_mask = tmin >= self.FREEZING_F
        rainfall[rain_mask] = gross_precip[rain_mask]

        # Mixed precipitation for temps straddling freezing point
        mixed_mask = ~snow_mask & ~rain_mask
        if np.any(mixed_mask):
            fraction_snow = ((self.FREEZING_F - tmin[mixed_mask]) /
                             (tmax[mixed_mask] - tmin[mixed_mask]))
            snowfall[mixed_mask] = gross_precip[mixed_mask] * fraction_snow
            rainfall[mixed_mask] = gross_precip[mixed_mask] * (1 - fraction_snow)

        # Calculate net values after interception
        net_snowfall = np.maximum(0.0, snowfall - interception)
        net_rainfall = np.maximum(0.0, rainfall - interception)

        return snowfall, net_snowfall, rainfall, net_rainfall

    def calculate_precipitation(self, date: datetime,
                                tmin: NDArray[np.float32],
                                tmax: NDArray[np.float32],
                                interception: NDArray[np.float32]) -> PrecipitationData:
        """Calculate precipitation components for current timestep"""
        # Ensure input arrays are properly shaped
        tmin = self._ensure_shape(tmin)
        tmax = self._ensure_shape(tmax)
        interception = self._ensure_shape(interception)

        # Get precipitation value based on method
        if self.use_fragments:
            precip = self._get_precipitation_from_fragments(date)
            if np.isscalar(precip):
                precip = np.full(self.domain_size, precip, dtype=np.float32)
            else:
                precip = self._ensure_shape(precip)
        else:
            try:
                precip = self.precip_reader.read_timestep(date)
                precip = self._ensure_shape(precip)
            except Exception as e:
                raise RuntimeError(f"Error reading precipitation for {date}: {str(e)}")

        # Initialize output arrays
        snowfall = np.zeros_like(precip)
        rainfall = np.zeros_like(precip)

        # Partition into rain and snow
        snow_mask = tmax <= self.FREEZING_F
        rain_mask = tmin >= self.FREEZING_F
        mixed_mask = ~snow_mask & ~rain_mask

        # All snow when max temp is below freezing
        snowfall[snow_mask] = precip[snow_mask]

        # All rain when min temp is above freezing
        rainfall[rain_mask] = precip[rain_mask]

        # Handle mixed precipitation
        if np.any(mixed_mask):
            fraction_snow = ((self.FREEZING_F - tmin[mixed_mask]) /
                             (tmax[mixed_mask] - tmin[mixed_mask]))
            snowfall[mixed_mask] = precip[mixed_mask] * fraction_snow
            rainfall[mixed_mask] = precip[mixed_mask] * (1 - fraction_snow)

        # Calculate net values after interception
        net_snowfall = np.maximum(0.0, snowfall - interception)
        net_rainfall = np.maximum(0.0, rainfall - interception)

        return PrecipitationData(
            date=date,
            gross_precip=precip,
            rainfall=rainfall,
            snowfall=snowfall,
            net_rainfall=net_rainfall,
            net_snowfall=net_snowfall
        )

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.precip_reader is not None:
            self.precip_reader.close()
            self.precip_reader = None

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

    def _get_precipitation_from_fragments(self, date: datetime) -> float:
        """Get precipitation value from fragments for current date"""
        month = date.month
        day = date.day
        
        if day == 1:
            self._update_fragments(True, month)
        else:
            self._update_fragments(False, month)
            
        precip = 0.0
        for zone in np.unique(self.rainfall_zones):
            if zone > 0:  # Skip no-data zones
                fragment = self._get_current_fragment(zone, month)
                if fragment:
                    precip += fragment['values'][day - 1]
                    
        return precip
    
    def _update_fragments(self, shuffle: bool, month: int) -> None:
        """Update fragment selection for current timestep"""
        if shuffle:
            self.random_values = np.random.random(self.random_values.shape)
            
        for zone in np.unique(self.rainfall_zones):
            if zone > 0:
                fragment_set = self.fragments_sets.get(zone)
                if fragment_set:
                    start = fragment_set.start_record[month]
                    count = fragment_set.number_of_fragments[month]
                    if count > 0:
                        idx = start + int(self.random_values[zone - 1] * count)
                        self.current_fragments[zone - 1] = self.fragments[idx]
    
    def _normalize_february_fragments(self, values: list) -> None:
        """Normalize February fragment values"""
        if values[28] > 0:  # Has day 29
            sum_28 = sum(values[:28])
            if sum_28 > 0:
                values[:28] = [v / sum_28 for v in values[:28]]
            values[28:] = [0.0] * (len(values) - 28)
    
    def _get_current_fragment(self, zone: int, month: int) -> Optional[Dict]:
        """Get current fragment for given zone and month"""
        if self.current_fragments is None or zone <= 0:
            return None
        return self.current_fragments[zone - 1]

