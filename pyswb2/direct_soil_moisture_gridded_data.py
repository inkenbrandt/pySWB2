
import numpy as np
import xarray as xr

class DirectSoilMoistureGriddedData:
    def __init__(self, grid_shape):
        """Initialize the gridded soil moisture data."""
        self.grid_shape = grid_shape
        self.soil_moisture = np.zeros(grid_shape, dtype=float)
        self.time = None

    def read_from_netcdf(self, filepath):
        """Read gridded soil moisture data from a NetCDF file."""
        ds = xr.open_dataset(filepath)
        self.soil_moisture = ds['soil_moisture'].values
        self.time = ds['time'].values
        ds.close()

    def write_to_netcdf(self, filepath):
        """Write gridded soil moisture data to a NetCDF file."""
        ds = xr.Dataset(
            {
                "soil_moisture": (("y", "x"), self.soil_moisture),
            },
            coords={
                "time": self.time,
            }
        )
        ds.to_netcdf(filepath)

    def update_soil_moisture(self, updates):
        """Update soil moisture grid with new values."""
        self.soil_moisture += updates


# Example Usage
if __name__ == "__main__":
    grid_shape = (100, 100)  # Example grid shape
    soil_moisture_data = DirectSoilMoistureGriddedData(grid_shape)
    soil_moisture_data.update_soil_moisture(np.random.random(grid_shape) * 0.1)
    print("Updated soil moisture:")
    print(soil_moisture_data.soil_moisture)
