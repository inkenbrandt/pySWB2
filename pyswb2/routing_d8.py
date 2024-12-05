
import numpy as np

class RoutingD8:
    def __init__(self, grid_shape):
        """Initialize the RoutingD8 class with the shape of the grid."""
        self.grid_shape = grid_shape
        self.target_row = np.full(grid_shape, -1, dtype=int)
        self.target_col = np.full(grid_shape, -1, dtype=int)
        self.is_downslope_target_marked = np.zeros(grid_shape, dtype=bool)
        self.sum_of_upslope_cells = np.zeros(grid_shape, dtype=int)
        self.number_of_upslope_connections = np.zeros(grid_shape, dtype=int)

    def initialize(self, d8_flow_direction):
        """Initialize the D8 flow direction grid."""
        self.d8_flow_direction = np.array(d8_flow_direction, dtype=int)

    def get_cell_index(self, row, col):
        """Get the index of a cell in a linear array (row-major order)."""
        return row * self.grid_shape[1] + col

    def get_target_index(self, row, col):
        """Get the target index for a given cell based on D8 flow direction."""
        if self.target_row[row, col] >= 0 and self.target_col[row, col] >= 0:
            return self.get_cell_index(self.target_row[row, col], self.target_col[row, col])
        else:
            return None

    def calculate_upslope_cells(self):
        """Calculate the sum of upslope cells for each grid cell."""
        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                target_row = self.target_row[row, col]
                target_col = self.target_col[row, col]
                if 0 <= target_row < self.grid_shape[0] and 0 <= target_col < self.grid_shape[1]:
                    self.sum_of_upslope_cells[target_row, target_col] += 1
                    self.number_of_upslope_connections[target_row, target_col] += 1

# Example Usage
if __name__ == "__main__":
    grid_shape = (4, 4)  # Example grid shape
    d8_flow_direction = np.random.randint(-1, 8, size=grid_shape)  # Example D8 directions

    routing = RoutingD8(grid_shape)
    routing.initialize(d8_flow_direction)

    # Example: Set targets for demonstration purposes
    routing.target_row[1, 1] = 2
    routing.target_col[1, 1] = 2

    # Calculate upslope cells
    routing.calculate_upslope_cells()

    print("Sum of Upslope Cells:")
    print(routing.sum_of_upslope_cells)
