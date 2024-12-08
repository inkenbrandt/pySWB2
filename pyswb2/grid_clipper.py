"""
Grid Clipper module for handling grid data alignment and format conversion.
Supports both ASCII and NetCDF grid formats.
"""

import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Union, Dict, Any
import sys
from dataclasses import dataclass
from scipy.ndimage import zoom
import netCDF4 as nc


@dataclass
class GridExtent:
    """Grid extent information"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    cell_size: float
    nx: int
    ny: int

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return bounds as (xmin, ymin, xmax, ymax)"""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def validate(self) -> None:
        """Validate grid extent parameters"""
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.cell_size <= 0:
            raise ValueError("Cell size must be positive")
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("Invalid extent bounds")


class GridFormat:
    """Enumeration of supported grid formats"""
    ASCII = "ascii"
    NETCDF = "netcdf"

    @staticmethod
    def detect_format(filepath: Path) -> str:
        """Detect grid format from file extension"""
        if filepath.suffix.lower() in ['.nc', '.nc4', '.netcdf']:
            return GridFormat.NETCDF
        elif filepath.suffix.lower() in ['.asc', '.txt']:
            return GridFormat.ASCII
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


class GridReader:
    """Abstract base class for grid readers"""

    def read(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def write(self, filepath: Path, data: np.ndarray, metadata: Dict[str, Any]) -> None:
        raise NotImplementedError


class ASCIIGridReader(GridReader):
    """Reader for ASCII grid format"""

    def read(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        header = {}
        with open(filepath) as f:
            for _ in range(6):
                key, val = f.readline().split()
                header[key.lower()] = float(val)
            data = np.loadtxt(f)
        return data, header

    def write(self, filepath: Path, data: np.ndarray, metadata: Dict[str, Any]) -> None:
        with open(filepath, 'w') as f:
            f.write(f"NCOLS {int(metadata['ncols'])}\n")
            f.write(f"NROWS {int(metadata['nrows'])}\n")
            f.write(f"XLLCORNER {metadata['xllcorner']:.6f}\n")
            f.write(f"YLLCORNER {metadata['yllcorner']:.6f}\n")
            f.write(f"CELLSIZE {metadata['cellsize']:.6f}\n")
            f.write(f"NODATA_VALUE {metadata['nodata_value']}\n")
            np.savetxt(f, data, fmt='%.6f')


class NetCDFGridReader(GridReader):
    """Reader for NetCDF grid format"""

    def read(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        with nc.Dataset(filepath, 'r') as ds:
            # Find the main variable
            var_name = None
            for var in ds.variables:
                if var not in ds.dimensions:
                    var_name = var
                    break

            if var_name is None:
                raise ValueError("No data variable found in NetCDF file")

            # Get data
            data = ds.variables[var_name][:]

            # Extract metadata
            metadata = {
                'ncols': data.shape[1],
                'nrows': data.shape[0],
                'xllcorner': float(ds.variables['x'][0]),
                'yllcorner': float(ds.variables['y'][0]),
                'cellsize': float(ds.variables['x'][1] - ds.variables['x'][0]),
                'nodata_value': ds.variables[var_name]._FillValue,
                'variable_name': var_name,
                'units': getattr(ds.variables[var_name], 'units', ''),
                'projection': getattr(ds, 'projection', '')
            }

            return data, metadata

    def write(self, filepath: Path, data: np.ndarray, metadata: Dict[str, Any]) -> None:
        with nc.Dataset(filepath, 'w', format='NETCDF4') as ds:
            # Create dimensions
            ds.createDimension('y', data.shape[0])
            ds.createDimension('x', data.shape[1])

            # Create coordinate variables
            x = ds.createVariable('x', 'f8', ('x',))
            y = ds.createVariable('y', 'f8', ('y',))

            # Set coordinate values
            x[:] = np.linspace(metadata['xllcorner'],
                               metadata['xllcorner'] + metadata['cellsize'] * metadata['ncols'],
                               metadata['ncols'])
            y[:] = np.linspace(metadata['yllcorner'],
                               metadata['yllcorner'] + metadata['cellsize'] * metadata['nrows'],
                               metadata['nrows'])

            # Create data variable
            var_name = metadata.get('variable_name', 'data')
            var = ds.createVariable(var_name, 'f4', ('y', 'x'),
                                    fill_value=metadata['nodata_value'])

            # Set data and attributes
            var[:] = data
            if 'units' in metadata:
                var.units = metadata['units']
            if 'projection' in metadata:
                ds.projection = metadata['projection']


class GridClipper:
    """Utility class to align grids to target extent and resolution"""

    def __init__(self, target_extent: GridExtent):
        """Initialize with target grid specifications

        Args:
            target_extent: Target grid extent and specifications
        """
        # Validate target extent
        target_extent.validate()
        self.target = target_extent

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Enable debug logging

        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

        # Initialize readers
        self.readers = {
            GridFormat.ASCII: ASCIIGridReader(),
            GridFormat.NETCDF: NetCDFGridReader()
        }

        # Log initialization
        self.logger.info(f"Initialized GridClipper with target grid: "
                         f"{self.target.nx}x{self.target.ny} cells, "
                         f"cell size: {self.target.cell_size}")

    def read_grid(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read grid file in any supported format"""
        format_type = GridFormat.detect_format(filepath)
        reader = self.readers[format_type]
        return reader.read(filepath)

    def write_grid(self, filepath: Path, data: np.ndarray,
                   metadata: Dict[str, Any], format_type: Optional[str] = None) -> None:
        """Write grid file in specified format"""
        if format_type is None:
            format_type = GridFormat.detect_format(filepath)
        reader = self.readers[format_type]
        reader.write(filepath, data, metadata)

    def resample_grid(self, data: np.ndarray, src_cell_size: float,
                      target_cell_size: float, method: str = 'nearest') -> np.ndarray:
        """Resample grid to target cell size"""
        if abs(src_cell_size - target_cell_size) < 1e-6:
            return data

        scale_factor = src_cell_size / target_cell_size

        if method == 'nearest':
            order = 0
        elif method == 'bilinear':
            order = 1
        elif method == 'cubic':
            order = 3
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        return zoom(data, scale_factor, order=order)

    def align_grid(self, data: np.ndarray, src_extent: GridExtent,
                   method: str = 'nearest', fill_value: float = -9999) -> np.ndarray:
        """Align source grid to target grid

        Args:
            data: Source grid data
            src_extent: Source grid extent information
            method: Resampling method ('nearest', 'bilinear', 'cubic')
            fill_value: Value to use for cells outside source extent

        Returns:
            Aligned grid matching target dimensions
        """
        # First handle resampling if needed
        if abs(src_extent.cell_size - self.target.cell_size) > 1e-6:
            self.logger.info(f"Resampling from {src_extent.cell_size} to {self.target.cell_size}")
            scale_x = self.target.nx / src_extent.nx
            scale_y = self.target.ny / src_extent.ny
            order = {'nearest': 0, 'bilinear': 1, 'cubic': 3}[method]
            data = zoom(data, (scale_y, scale_x), order=order)

            # Update source extent after resampling
            src_extent = GridExtent(
                xmin=src_extent.xmin,
                ymin=src_extent.ymin,
                xmax=src_extent.xmax,
                ymax=src_extent.ymax,
                cell_size=self.target.cell_size,
                nx=self.target.nx,
                ny=self.target.ny
            )

        # Create output grid
        aligned = np.full((self.target.ny, self.target.nx), fill_value, dtype=data.dtype)

        # Calculate grid coordinates
        src_x = np.linspace(src_extent.xmin, src_extent.xmax, src_extent.nx)
        src_y = np.linspace(src_extent.ymin, src_extent.ymax, src_extent.ny)
        dst_x = np.linspace(self.target.xmin, self.target.xmax, self.target.nx)
        dst_y = np.linspace(self.target.ymin, self.target.ymax, self.target.ny)

        # Find overlapping region
        x_min = max(src_extent.xmin, self.target.xmin)
        x_max = min(src_extent.xmax, self.target.xmax)
        y_min = max(src_extent.ymin, self.target.ymin)
        y_max = min(src_extent.ymax, self.target.ymax)

        # Calculate indices in both grids
        src_x_idx = np.where((src_x >= x_min) & (src_x <= x_max))[0]
        src_y_idx = np.where((src_y >= y_min) & (src_y <= y_max))[0]
        dst_x_idx = np.where((dst_x >= x_min) & (dst_x <= x_max))[0]
        dst_y_idx = np.where((dst_y >= y_min) & (dst_y <= y_max))[0]

        if len(src_x_idx) == 0 or len(src_y_idx) == 0 or len(dst_x_idx) == 0 or len(dst_y_idx) == 0:
            self.logger.warning("No overlap between source and target grids")
            return aligned

        # Get slice ranges
        src_i_start, src_i_end = src_y_idx[0], src_y_idx[-1] + 1
        src_j_start, src_j_end = src_x_idx[0], src_x_idx[-1] + 1
        dst_i_start, dst_i_end = dst_y_idx[0], dst_y_idx[-1] + 1
        dst_j_start, dst_j_end = dst_x_idx[0], dst_x_idx[-1] + 1

        # Debug output
        self.logger.debug(f"Source slice: [{src_i_start}:{src_i_end}, {src_j_start}:{src_j_end}]")
        self.logger.debug(f"Target slice: [{dst_i_start}:{dst_i_end}, {dst_j_start}:{dst_j_end}]")
        self.logger.debug(f"Source shape: {data[src_i_start:src_i_end, src_j_start:src_j_end].shape}")
        self.logger.debug(f"Target shape: {aligned[dst_i_start:dst_i_end, dst_j_start:dst_j_end].shape}")

        # Ensure slices match in size
        width = min(src_j_end - src_j_start, dst_j_end - dst_j_start)
        height = min(src_i_end - src_i_start, dst_i_end - dst_i_start)

        # Copy data
        aligned[dst_i_start:dst_i_start + height, dst_j_start:dst_j_start + width] = \
            data[src_i_start:src_i_start + height, src_j_start:src_j_start + width]

        return aligned

    def process_file(self, input_path: Path, output_path: Path,
                     method: str = 'nearest', fill_value: float = -9999) -> None:
        """Process a single input file"""
        try:
            # Read input
            data, metadata = self.read_grid(input_path)

            # Create source extent
            src_extent = GridExtent(
                xmin=float(metadata['xllcorner']),
                ymin=float(metadata['yllcorner']),
                xmax=float(metadata['xllcorner'] + metadata['ncols'] * metadata['cellsize']),
                ymax=float(metadata['yllcorner'] + metadata['nrows'] * metadata['cellsize']),
                cell_size=float(metadata['cellsize']),
                nx=int(metadata['ncols']),
                ny=int(metadata['nrows'])
            )

            # Log alignment info
            self.logger.info(f"Processing {input_path.name}")
            self.logger.info(f"Source grid: {src_extent.nx}x{src_extent.ny}, cell size: {src_extent.cell_size}")
            self.logger.info(f"Target grid: {self.target.nx}x{self.target.ny}, cell size: {self.target.cell_size}")

            # Align grid
            aligned = self.align_grid(data, src_extent, method, fill_value)

            # Update metadata for output
            metadata.update({
                'ncols': self.target.nx,
                'nrows': self.target.ny,
                'xllcorner': self.target.xmin,
                'yllcorner': self.target.ymin,
                'cellsize': self.target.cell_size,
                'nodata_value': fill_value
            })

            # Write output
            self.write_grid(output_path, aligned, metadata)
            self.logger.info(f"Wrote aligned grid to {output_path}")

        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            raise


def setup_logging(log_file: Optional[Path] = None) -> None:
    """Set up logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """Command line interface for grid clipping"""
    import argparse

    parser = argparse.ArgumentParser(description="Clip and align grid files to target specifications")
    parser.add_argument("input_dir", type=Path, help="Input directory containing grid files")
    parser.add_argument("output_dir", type=Path, help="Output directory for clipped files")
    parser.add_argument("--target", type=Path, required=True, help="Target grid file to match")
    parser.add_argument("--pattern", default="*.asc", help="File pattern to process")
    parser.add_argument("--method", choices=['nearest', 'bilinear', 'cubic'],
                        default='nearest', help="Resampling method")
    parser.add_argument("--fill", type=float, default=-9999, help="Fill value for nodata cells")
    parser.add_argument("--log", type=Path, help="Log file path")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log)
    logger = logging.getLogger(__name__)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read target grid info to get extents
        clipper = None
        with open(args.target) as f:
            header = {}
            for _ in range(6):
                key, val = f.readline().split()
                header[key.lower()] = float(val)

        target_extent = GridExtent(
            xmin=float(header['xllcorner']),
            ymin=float(header['yllcorner']),
            xmax=float(header['xllcorner'] + header['ncols'] * header['cellsize']),
            ymax=float(header['yllcorner'] + header['nrows'] * header['cellsize']),
            cell_size=float(header['cellsize']),
            nx=int(header['ncols']),
            ny=int(header['nrows'])
        )

        # Initialize clipper
        clipper = GridClipper(target_extent)

        # Process files
        for input_path in args.input_dir.glob(args.pattern):
            try:
                output_path = args.output_dir / input_path.name
                clipper.process_file(input_path, output_path, args.method, args.fill)
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {str(e)}")
    except Exception as e:
        logger.error(f"{str(e)}")
