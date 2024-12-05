from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import netCDF4 as nc

@dataclass
class NetCDFDimension:
    name: str
    dim_id: int = -9999
    size: int = 0
    unlimited: bool = False

@dataclass
class NetCDFAttribute:
    name: str
    values: Union[List[str], List[int], List[float], np.ndarray]
    dtype: str
    size: int

@dataclass
class NetCDFVariable:
    name: str
    var_id: int = -9999
    var_type: str = ''
    dimensions: List[int] = None
    attributes: List[NetCDFAttribute] = None
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = []
        if self.attributes is None:
            self.attributes = []

class NetCDF4File:
    def __init__(self):
        self.ncid: Optional[nc.Dataset] = None
        self.filename: str = ''
        self.file_format: str = ''
        self.dimensions: List[NetCDFDimension] = []
        self.variables: List[NetCDFVariable] = []
        self.attributes: List[NetCDFAttribute] = []
        
        # Grid properties
        self.nx: int = 0
        self.ny: int = 0
        self.dx: float = 0.0
        self.x_coords: Optional[np.ndarray] = None
        self.y_coords: Optional[np.ndarray] = None
        
        # Time properties
        self.origin_date: Optional[datetime] = None
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.time_values: Optional[np.ndarray] = None
        
        # Data handling
        self.scale_factor: float = 1.0
        self.add_offset: float = 0.0
        self.flip_horizontal: bool = False
        self.flip_vertical: bool = False
        
    def open(self, filename: str, mode: str = 'r') -> None:
        """Open netCDF file"""
        self.filename = filename
        self.ncid = nc.Dataset(filename, mode)
        self._populate_metadata()
        
    def close(self) -> None:
        """Close netCDF file"""
        if self.ncid:
            self.ncid.close()
            
    def _populate_metadata(self) -> None:
        """Populate metadata from opened file"""
        # Dimensions
        for name, dim in self.ncid.dimensions.items():
            self.dimensions.append(NetCDFDimension(
                name=name,
                dim_id=len(self.dimensions),
                size=len(dim),
                unlimited=dim.isunlimited()
            ))
            
        # Variables
        for name, var in self.ncid.variables.items():
            variable = NetCDFVariable(
                name=name,
                var_id=len(self.variables),
                var_type=var.dtype.name,
                dimensions=[dim.dim_id for dim in self.dimensions if dim.name in var.dimensions]
            )
            
            # Variable attributes
            for attr_name in var.ncattrs():
                attr_value = var.getncattr(attr_name)
                if isinstance(attr_value, str):
                    attr_value = [attr_value]
                variable.attributes.append(NetCDFAttribute(
                    name=attr_name,
                    values=attr_value,
                    dtype=type(attr_value[0]).__name__,
                    size=len(attr_value) if isinstance(attr_value, (list, np.ndarray)) else 1
                ))
            self.variables.append(variable)
            
        # Global attributes
        for attr_name in self.ncid.ncattrs():
            attr_value = self.ncid.getncattr(attr_name)
            if isinstance(attr_value, str):
                attr_value = [attr_value]
            self.attributes.append(NetCDFAttribute(
                name=attr_name,
                values=attr_value,
                dtype=type(attr_value[0]).__name__,
                size=len(attr_value) if isinstance(attr_value, (list, np.ndarray)) else 1
            ))

    def get_variable(self, name: str, start: Optional[List[int]] = None,
                    count: Optional[List[int]] = None,
                    stride: Optional[List[int]] = None) -> np.ndarray:
        """Get variable data with optional slicing"""
        var = self.ncid.variables[name]
        return var[tuple(slice(s, s+c, st) for s, c, st in zip(start or [None]*var.ndim,
                                                              count or [None]*var.ndim,
                                                              stride or [None]*var.ndim))]

    def put_variable(self, name: str, data: np.ndarray,
                    start: Optional[List[int]] = None,
                    count: Optional[List[int]] = None) -> None:
        """Put variable data with optional slicing"""
        var = self.ncid.variables[name]
        if start is not None and count is not None:
            slices = tuple(slice(s, s+c) for s, c in zip(start, count))
            var[slices] = data
        else:
            var[:] = data

    def create_variable(self, name: str, datatype: str, dimensions: List[str],
                       fill_value: Optional[Union[int, float]] = None,
                       compression: bool = True) -> None:
        """Create a new variable"""
        if compression:
            self.ncid.createVariable(name, datatype, dimensions,
                                   fill_value=fill_value,
                                   zlib=True, complevel=4)
        else:
            self.ncid.createVariable(name, datatype, dimensions,
                                   fill_value=fill_value)

    def set_attribute(self, var_name: str, attr_name: str, value: Union[str, int, float, np.ndarray]) -> None:
        """Set attribute for variable or globally"""
        if var_name == 'global':
            self.ncid.setncattr(attr_name, value)
        else:
            self.ncid.variables[var_name].setncattr(attr_name, value)
