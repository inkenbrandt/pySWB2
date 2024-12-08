from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np
import netCDF4 as nc
from enum import Enum


@dataclass
class GridData:
    """Container for grid data with metadata"""
    data: np.ndarray
    metadata: Dict[str, Union[str, float, int]]
    transform: Optional[Dict[str, float]] = None

    def validate(self) -> None:
        """Validate grid data and metadata"""
        if self.data is None:
            raise ValueError("Grid data cannot be None")
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        if self.metadata is None:
            raise ValueError("Metadata cannot be None")


class DataFormat(Enum):
    """Supported data formats"""
    ASCII = ".asc"
    NETCDF = ".nc"
    GEOTIFF = ".tif"


class DataInterface:
    """Base class for data format handlers"""
    
    def read(self, path: Path) -> GridData:
        raise NotImplementedError
        
    def write(self, data: GridData, path: Path) -> None:
        raise NotImplementedError


class ASCIIGridInterface(DataInterface):
    """Handler for ASCII grid format"""
    
    def read(self, path: Path) -> GridData:
        """Read ASCII grid file"""
        header = {}
        data = None
        
        with open(path) as f:
            # Read header
            for _ in range(6):
                key, val = f.readline().split()
                header[key.lower()] = float(val)
            
            # Read data
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
        
    def write(self, data: GridData, path: Path) -> None:
        """Write ASCII grid file"""
        with open(path, 'w') as f:
            # Write header
            f.write(f"NCOLS {data.metadata['nx']}\n")
            f.write(f"NROWS {data.metadata['ny']}\n")
            f.write(f"XLLCORNER {data.metadata['x_ll']}\n")
            f.write(f"YLLCORNER {data.metadata['y_ll']}\n")
            f.write(f"CELLSIZE {data.metadata['cell_size']}\n")
            f.write(f"NODATA_VALUE {data.metadata['nodata']}\n")
            
            # Write data
            np.savetxt(f, data.data, fmt='%.3f')


class NetCDFInterface(DataInterface):
    """Handler for NetCDF format"""
    
    def read(self, path: Path) -> GridData:
        """Read NetCDF file"""
        with nc.Dataset(path, 'r') as ds:
            # Get main variable
            var_name = next(iter(ds.variables))  # First variable
            var = ds.variables[var_name]
            
            # Read data and metadata
            data = var[:]
            metadata = {
                'units': getattr(var, 'units', ''),
                'description': getattr(var, 'description', ''),
                'nodata': getattr(var, '_FillValue', -9999.0)
            }
            
            # Add dimension info if available
            if hasattr(ds, 'nx'):
                metadata.update({
                    'nx': ds.nx,
                    'ny': ds.ny,
                    'x_ll': getattr(ds, 'x_ll', 0.0),
                    'y_ll': getattr(ds, 'y_ll', 0.0),
                    'cell_size': getattr(ds, 'cell_size', 1.0)
                })
                
        return GridData(data=data, metadata=metadata)
        
    def write(self, data: GridData, path: Path) -> None:
        """Write NetCDF file"""
        with nc.Dataset(path, 'w') as ds:
            # Create dimensions
            ds.createDimension('y', data.data.shape[0])
            ds.createDimension('x', data.data.shape[1])
            
            # Create variable
            var = ds.createVariable('data', 'f4', ('y', 'x'),
                                  fill_value=data.metadata.get('nodata', -9999.0))
            
            # Write data
            var[:] = data.data
            
            # Add metadata
            for key, value in data.metadata.items():
                setattr(var, key, value)


class DataManager:
    """Manager class for handling different data formats"""
    
    def __init__(self):
        """Initialize with supported formats"""
        self.interfaces: Dict[str, DataInterface] = {
            DataFormat.ASCII.value: ASCIIGridInterface(),
            DataFormat.NETCDF.value: NetCDFInterface()
        }
        
        # Cache for loaded data
        self.cache: Dict[str, GridData] = {}
        
    def load(self, path: Path, transform: Optional[Dict[str, float]] = None) -> GridData:
        """Load data from file with optional transform"""
        # Return cached data if available
        cache_key = str(path)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get appropriate interface
        interface = self.interfaces.get(path.suffix.lower())
        if not interface:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        # Load data
        try:
            data = interface.read(path)
        except Exception as e:
            raise IOError(f"Failed to read {path}: {str(e)}")
            
        # Apply transform if provided
        if transform:
            data.transform = transform
            data.data = self._apply_transform(data.data, transform)
            
        # Cache and return
        self.cache[cache_key] = data
        return data
        
    def write(self, data: Union[GridData, Dict[str, np.ndarray]], path: Path) -> None:
        """Write data to file"""
        # Convert dict to GridData if needed
        if isinstance(data, dict):
            data = self._convert_dict_to_griddata(data)
            
        # Validate data
        data.validate()
        
        # Get appropriate interface
        interface = self.interfaces.get(path.suffix.lower())
        if not interface:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        # Write data
        try:
            interface.write(data, path)
        except Exception as e:
            raise IOError(f"Failed to write {path}: {str(e)}")
            
    def clear_cache(self) -> None:
        """Clear data cache"""
        self.cache.clear()
        
    def _apply_transform(self, data: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
        """Apply transform to data"""
        result = data.copy()
        
        if 'scale' in transform:
            result *= transform['scale']
        if 'offset' in transform:
            result += transform['offset']
        if 'missing_value' in transform:
            result[result == transform['missing_value']] = np.nan
            
        return result
        
    def _convert_dict_to_griddata(self, data_dict: Dict[str, np.ndarray]) -> GridData:
        """Convert dictionary of arrays to GridData"""
        # Use first array for basic metadata
        first_array = next(iter(data_dict.values()))
        
        return GridData(
            data=first_array,
            metadata={
                'nx': first_array.shape[1],
                'ny': first_array.shape[0],
                'nodata': -9999.0
            }
        )
