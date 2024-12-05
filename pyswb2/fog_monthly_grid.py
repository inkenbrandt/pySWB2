class FogMonthlyGrid:
    """
    Python translation of the Fortran module fog__monthly_grid.
    Handles fog drip estimation using gridded fog ratio and capture efficiency.
    """
    def __init__(self, grid_shape):
        """
        Initializes the fog drip calculation structure.

        Parameters:
        - grid_shape (tuple): Shape of the 2D grid (rows, cols).
        """
        self.grid_shape = grid_shape
        self.fog_ratio = [[0.0 for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]  # Fog ratio grid
        self.fog_capture_efficiency = [[0.0 for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]  # Capture efficiency grid
        self.fog_drip = [[0.0 for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]  # Calculated fog drip

    def initialize_grid(self, fog_ratio_values, capture_efficiency_values):
        """
        Initializes the grid values for fog ratio and capture efficiency.

        Parameters:
        - fog_ratio_values (list of lists): 2D grid of fog ratio values.
        - capture_efficiency_values (list of lists): 2D grid of capture efficiency values.
        """
        self.fog_ratio = fog_ratio_values
        self.fog_capture_efficiency = capture_efficiency_values

    def calculate_fog_drip(self, active_cells):
        """
        Calculates fog drip for active cells in the grid.

        Parameters:
        - active_cells (list of lists): 2D boolean grid indicating active cells.

        Returns:
        - fog_drip (list of lists): Updated 2D grid of fog drip values.
        """
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if active_cells[i][j]:
                    self.fog_drip[i][j] = self.fog_ratio[i][j] * self.fog_capture_efficiency[i][j]
                else:
                    self.fog_drip[i][j] = 0.0
        return self.fog_drip

    def display_fog_drip(self):
        """
        Displays the calculated fog drip grid for debugging purposes.
        """
        print("Fog Drip Grid:")
        for row in self.fog_drip:
            print(row)

if __name__ == "__main__":
    # Example usage
    grid_shape = (3, 3)
    fog_model = FogMonthlyGrid(grid_shape)

    # Initialize fog ratio and capture efficiency grids
    fog_ratio_values = [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5]]
    capture_efficiency_values = [[0.8, 0.9, 1.0], [0.7, 0.8, 0.9], [0.6, 0.7, 0.8]]
    fog_model.initialize_grid(fog_ratio_values, capture_efficiency_values)

    # Define active cells
    active_cells = [[True, False, True], [True, True, False], [False, True, True]]

    # Calculate fog drip
    fog_drip = fog_model.calculate_fog_drip(active_cells)

    # Display results
    fog_model.display_fog_drip()
