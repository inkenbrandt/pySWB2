
import numpy as np

class ContinuousFrozenGroundIndex:
    """Class to manage Continuous Frozen Ground Index (CFGI) calculations."""

    def __init__(self, grid_shape):
        """Initialize CFGI data structures.

        Args:
            grid_shape (tuple): Shape of the grid (rows, cols).
        """
        self.cfgi = np.zeros(grid_shape, dtype=np.float32)  # Continuous frozen ground index
        self.cfgi_ll = np.zeros(grid_shape, dtype=np.float32)  # Lower limit of CFGI
        self.cfgi_ul = np.zeros(grid_shape, dtype=np.float32)  # Upper limit of CFGI
        self.active_cells = np.zeros(grid_shape, dtype=bool)  # Active cells in the grid

    def initialize(self, initial_cfgi, lower_limit, upper_limit, active_cells):
        """Initialize CFGI values and parameters.

        Args:
            initial_cfgi (array-like): Initial CFGI values.
            lower_limit (array-like): Lower limit of CFGI.
            upper_limit (array-like): Upper limit of CFGI.
            active_cells (array-like): Boolean array of active cells.
        """
        self.cfgi = np.array(initial_cfgi, dtype=np.float32)
        self.cfgi_ll = np.array(lower_limit, dtype=np.float32)
        self.cfgi_ul = np.array(upper_limit, dtype=np.float32)
        self.active_cells = np.array(active_cells, dtype=bool)
        print("CFGI initialization completed.")

    def update(self, cfgi_change):
        """Update CFGI values based on changes.

        Args:
            cfgi_change (array-like): Changes to apply to the CFGI.
        """
        change_array = np.array(cfgi_change, dtype=np.float32)
        self.cfgi[self.active_cells] += change_array[self.active_cells]
        self.cfgi = np.clip(self.cfgi, self.cfgi_ll, self.cfgi_ul)  # Enforce limits
        print("CFGI updated based on changes.")
