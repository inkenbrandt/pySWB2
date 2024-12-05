from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

class GridDataType(Enum):
    INTEGER = 0
    REAL = 1
    DOUBLE = 2

class OutputFormat(Enum):
    SURFER = 0 
    ARC = 1

@dataclass
class GridBounds:
    """Grid boundary coordinates"""
    proj4_string: str
    nx: int 
    ny: int
    x_ll: float  # Lower left X
    y_ll: float  # Lower left Y
    x_ur: float  # Upper right X 
    y_ur: float  # Upper right Y
    cell_size: float

class Grid:
    """Grid data structure and operations"""
    
    def __init__(self, 
                 nx: int,
                 ny: int, 
                 x0: float,
                 y0: float,
                 x1: float = None,
                 y1: float = None,
                 cell_size: float = None,
                 data_type: GridDataType = GridDataType.REAL,
                 proj4_string: str = None):
        
        self.nx = nx
        self.ny = ny
        self.x0 = x0  # Lower left
        self.y0 = y0
        self.data_type = data_type
        self.proj4_string = proj4_string
        
        # Calculate grid extents
        if cell_size is not None:
            self.cell_size = cell_size
            self.x1 = x0 + nx * cell_size
            self.y1 = y0 + ny * cell_size
        elif x1 is not None and y1 is not None:
            self.x1 = x1
            self.y1 = y1
            self.cell_size = (x1 - x0) / nx
        else:
            raise ValueError("Must provide either cell_size or x1,y1")

        # Initialize data arrays based on type
        if data_type == GridDataType.INTEGER:
            self.data = np.zeros((ny, nx), dtype=np.int32)
            self.nodata = -9999
        else:
            self.data = np.zeros((ny, nx), dtype=np.float32) 
            self.nodata = -9999.0
            
        self.mask = np.ones((ny, nx), dtype=bool)
        
        # Coordinate arrays
        self.x = None
        self.y = None
        
    def populate_xy(self) -> None:
        """Populate X,Y coordinate arrays"""
        if self.x is None:
            x = np.linspace(self.x0 + self.cell_size/2, 
                          self.x1 - self.cell_size/2, 
                          self.nx)
            y = np.linspace(self.y0 + self.cell_size/2,
                          self.y1 - self.cell_size/2,
                          self.ny)
            self.x, self.y = np.meshgrid(x, y)

    def get_row_col(self, x: float, y: float) -> Tuple[int, int]:
        """Get row,col indices for given x,y coordinates"""
        col = int((x - self.x0) / self.cell_size)
        row = int((self.y1 - y) / self.cell_size) 
        
        # Validate bounds
        if not (0 <= col < self.nx and 0 <= row < self.ny):
            return -1, -1
            
        return row, col

    def read_arc_grid(self, filename: str) -> None:
        """Read ARC ASCII grid file"""
        with open(filename) as f:
            # Read header
            header = {}
            for _ in range(6):
                key, val = f.readline().split()
                header[key.lower()] = float(val)
                
            # Read data
            data = np.loadtxt(f, dtype=self.data.dtype)
            self.data = data.reshape(self.ny, self.nx)
            
            # Set properties
            self.x0 = header['xllcorner']
            self.y0 = header['yllcorner']
            self.cell_size = header['cellsize']
            self.nodata = header['nodata_value']

    def write_arc_grid(self, filename: str) -> None:
        """Write to ARC ASCII grid format"""
        with open(filename, 'w') as f:
            f.write(f"NCOLS {self.nx}\n")
            f.write(f"NROWS {self.ny}\n")
            f.write(f"XLLCORNER {self.x0}\n")
            f.write(f"YLLCORNER {self.y0}\n") 
            f.write(f"CELLSIZE {self.cell_size}\n")
            f.write(f"NODATA_VALUE {self.nodata}\n")
            np.savetxt(f, self.data, fmt='%.3f')

    def interpolate(self, x: float, y: float) -> float:
        """Bilinear interpolation at x,y point"""
        # Get bounding cell indices
        row, col = self.get_row_col(x, y)
        if row < 0 or col < 0:
            return self.nodata
            
        # Calculate interpolation weights
        x_frac = (x - (self.x0 + col*self.cell_size)) / self.cell_size
        y_frac = (y - (self.y0 + row*self.cell_size)) / self.cell_size
        
        # Get corner values
        v00 = self.data[row,col]
        v01 = self.data[row,col+1] if col < self.nx-1 else v00
        v10 = self.data[row+1,col] if row < self.ny-1 else v00
        v11 = self.data[row+1,col+1] if row < self.ny-1 and col < self.nx-1 else v00
        
        # Bilinear interpolation
        return (v00 * (1-x_frac) * (1-y_frac) + 
                v01 * x_frac * (1-y_frac) +
                v10 * (1-x_frac) * y_frac +
                v11 * x_frac * y_frac)
