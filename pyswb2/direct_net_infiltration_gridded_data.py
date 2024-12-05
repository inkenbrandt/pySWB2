
import numpy as np
import xarray as xr

class DirectNetInfiltrationGriddedData:
    def __init__(self, grid_shape):
        """Initialize the gridded net infiltration data."""
        self.grid_shape = grid_shape
        self.net_infiltration = np.zeros(grid_shape, dtype=float)
        self.time = None

    def read_from_netcdf(self, filepath):
        """Read gridded net infiltration data from a NetCDF file."""
        ds = xr.open_dataset(filepath)
        self.net_infiltration = ds['net_infiltration'].values
        self.time = ds['time'].values
        ds.close()

    def write_to_netcdf(self, filepath):
        """Write gridded net infiltration data to a NetCDF file."""
        ds = xr.Dataset(
            {
                "net_infiltration": (("y", "x"), self.net_infiltration),
            },
            coords={
                "time": self.time,
            }
        )
        ds.to_netcdf(filepath)

    def update_net_infiltration(self, updates):
        """Update net infiltration grid with new values."""
        self.net_infiltration += updates


# Example Usage
if __name__ == "__main__":
    grid_shape = (100, 100)  # Example grid shape
    net_infiltration_data = DirectNetInfiltrationGriddedData(grid_shape)
    net_infiltration_data.update_net_infiltration(np.random.random(grid_shape) * 0.05)
    print("Updated net infiltration:")
    print(net_infiltration_data.net_infiltration)
