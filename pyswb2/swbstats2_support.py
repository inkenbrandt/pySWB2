
import numpy as np
from datetime import datetime

class Grid:
    def __init__(self, datatype, shape):
        self.datatype = datatype
        self.data = np.zeros(shape, dtype=datatype)

    def write_arc_grid(self, filename):
        """Simulate writing an ArcGrid file."""
        np.savetxt(filename, self.data, fmt='%f')

class SWBDateTime:
    def __init__(self, year, month, day):
        self.date = datetime(year, month, day)

    def __gt__(self, other):
        return self.date > other.date

class SWBStats2Support:
    def __init__(self):
        self.grids = []

    def create_grid(self, shape, datatype=float):
        grid = Grid(datatype, shape)
        self.grids.append(grid)
        return grid

    def assert_condition(self, condition, message):
        if not condition:
            raise AssertionError(message)
