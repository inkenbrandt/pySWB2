
import numpy as np

class AWCDepthIntegrated:
    """Class to manage depth-integrated Available Water Content (AWC)."""

    def __init__(self):
        """Initialize the depth-integrated AWC manager."""
        self.available_water_content = None  # Placeholder for AWC grid

    def initialize(self, grid_shape):
        """Initialize the grid structure for AWC.

        Args:
            grid_shape (tuple): Shape of the AWC grid (rows, cols).
        """
        self.available_water_content = np.zeros(grid_shape, dtype=np.float32)
        print(f"Initialized AWC grid with shape {grid_shape}.")

    def read(self, rooting_depth, soil_horizons):
        """Read and depth-average AWC data across soil horizons.

        Args:
            rooting_depth (array-like): Rooting depth for the soil grid (rows, cols).
            soil_horizons (list of array-like): List of AWC grids for different soil horizons.

        Raises:
            ValueError: If the grid shapes do not match.
        """
        rooting_depth_array = np.array(rooting_depth, dtype=np.float32)
        horizon_arrays = [np.array(h, dtype=np.float32) for h in soil_horizons]

        if self.available_water_content is None:
            raise ValueError("AWC grid must be initialized before reading data.")

        if rooting_depth_array.shape != self.available_water_content.shape:
            raise ValueError("Rooting depth grid shape does not match initialized AWC grid shape.")

        for h in horizon_arrays:
            if h.shape != self.available_water_content.shape:
                raise ValueError("All soil horizon grids must match the AWC grid shape.")

        # Calculate depth-integrated AWC (placeholder logic: average across horizons)
        depth_weighted_sum = sum(horizon_arrays)
        self.available_water_content = depth_weighted_sum / len(horizon_arrays)

        print("AWC depth-averaged and grid updated successfully.")

    def get_awc_value(self, row, col):
        """Retrieve the AWC value at a specific grid cell.

        Args:
            row (int): Row index.
            col (int): Column index.

        Returns:
            float: The AWC value at the specified grid cell.

        Raises:
            IndexError: If the indices are out of bounds.
        """
        if self.available_water_content is None:
            raise ValueError("AWC grid is not initialized.")
        return self.available_water_content[row, col]
