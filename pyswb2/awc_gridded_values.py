
import numpy as np

class AWCGriddedValues:
    """Class to manage Available Water Content (AWC) gridded values."""

    def __init__(self):
        """Initialize the AWC grid manager."""
        self.awc_grid = None  # Placeholder for gridded AWC data

    def initialize(self, grid_shape):
        """Initialize the grid structure for AWC.

        Args:
            grid_shape (tuple): Shape of the AWC grid (rows, cols).
        """
        self.awc_grid = np.zeros(grid_shape, dtype=np.float32)
        print(f"Initialized AWC grid with shape {grid_shape}.")

    def read_awc_grid(self, grid_data):
        """Read and populate the AWC grid.

        Args:
            grid_data (array-like): Input AWC gridded data.

        Raises:
            ValueError: If the input grid data is not compatible.
        """
        grid_array = np.array(grid_data, dtype=np.float32)
        if self.awc_grid is None:
            raise ValueError("AWC grid must be initialized before reading data.")
        if grid_array.shape != self.awc_grid.shape:
            raise ValueError("Input grid shape does not match initialized AWC grid shape.")
        self.awc_grid = grid_array
        print("AWC grid populated successfully.")

    def find_awc_value(self, row, col):
        """Retrieve an AWC value at a specific grid cell.

        Args:
            row (int): Row index.
            col (int): Column index.

        Returns:
            float: The AWC value at the specified grid cell.

        Raises:
            IndexError: If the indices are out of bounds.
        """
        if self.awc_grid is None:
            raise ValueError("AWC grid is not initialized.")
        return self.awc_grid[row, col]
