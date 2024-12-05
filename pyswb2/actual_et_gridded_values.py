
import numpy as np
from datetime import datetime

class GriddedActualET:
    """Class to manage gridded actual evapotranspiration (ET) values."""

    def __init__(self, grid_shape):
        """Initialize the ET grid manager.

        Args:
            grid_shape (tuple): Shape of the ET grid (rows, cols).
        """
        self.actual_et = np.zeros(grid_shape, dtype=np.float32)  # Array for actual ET values
        self.date_of_last_retrieval = None  # Tracks the last retrieval date
        print(f"Initialized ET grid with shape {grid_shape}.")

    def initialize(self, initial_values=None):
        """Initialize ET values with an optional set of values.

        Args:
            initial_values (array-like, optional): Initial ET values to populate the grid.
        """
        if initial_values is not None:
            initial_array = np.array(initial_values, dtype=np.float32)
            if initial_array.shape != self.actual_et.shape:
                raise ValueError("Initial values shape does not match ET grid shape.")
            self.actual_et = initial_array
        print("ET grid initialized.")

    def calculate(self, et_grid):
        """Calculate daily ET by substituting values from the provided ET grid.

        Args:
            et_grid (array-like): Grid of ET values for the current day.

        Returns:
            np.ndarray: Updated ET grid for the day.

        Raises:
            ValueError: If the input ET grid shape does not match.
        """
        et_array = np.array(et_grid, dtype=np.float32)
        if et_array.shape != self.actual_et.shape:
            raise ValueError("ET grid shape does not match initialized ET grid shape.")
        self.actual_et = et_array
        self.date_of_last_retrieval = datetime.now()
        print("Daily ET grid updated.")
        return self.actual_et

    def get_et_value(self, row, col):
        """Retrieve the ET value at a specific grid cell.

        Args:
            row (int): Row index.
            col (int): Column index.

        Returns:
            float: The ET value at the specified grid cell.

        Raises:
            IndexError: If the indices are out of bounds.
        """
        if self.actual_et is None:
            raise ValueError("ET grid is not initialized.")
        return self.actual_et[row, col]
