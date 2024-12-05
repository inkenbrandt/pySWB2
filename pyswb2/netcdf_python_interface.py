
import netCDF4 as nc
import numpy as np

class NetCDFInterface:
    def __init__(self, filepath):
        """Initialize the NetCDF interface with a file."""
        self.filepath = filepath

    def nc_get_vars_short(self, var_name, start_indices, count, stride):
        """
        Mimics the behavior of the Fortran nc_get_vars_short function using Python's NetCDF4 library.
        
        Args:
            var_name (str): Name of the variable to read.
            start_indices (list of int): Starting indices for reading.
            count (list of int): Number of values to read along each dimension.
            stride (list of int): Stride (step) for reading values.
            
        Returns:
            np.ndarray: The extracted data as a NumPy array.
        """
        with nc.Dataset(self.filepath, 'r') as dataset:
            variable = dataset.variables[var_name]
            slices = tuple(slice(start, start + n * step, step) 
                           for start, n, step in zip(start_indices, count, stride))
            data = variable[slices]
        return np.array(data)
