# Importing necessary Python libraries
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

@dataclass
class Grid:
    """
    Placeholder class to mimic the grid creation functionality in Fortran.
    Attributes and methods should be defined as required for the actual use case.
    """
    NX: int
    NY: int
    X0: float
    Y0: float
    X1: float
    Y1: float
    data_type: str

class RunningStats:
    """
    Python translation of RUNNING_STATS_T with statistical methods.
    """

    def __init__(self, nodata_value=-9999.0):
        self.grd_delta = None
        self.grd_delta_n = None
        self.grd_delta_n2 = None
        self.grd_term1 = None
        self.grd_M1 = None
        self.grd_M2 = None
        self.grd_sum = None
        self.n = 0
        self.nodata_value = nodata_value

    def initialize(self, NX, NY, X0, Y0, X1, Y1, nodata_value):
        """
        Initializes the RunningStats object with grid dimensions and bounds.
        """
        self.nodata_value = nodata_value
        self.grd_delta = Grid(NX=NX, NY=NY, X0=X0, Y0=Y0, X1=X1, Y1=Y1, data_type="double")
        self.grd_delta_n = Grid(NX=NX, NY=NY, X0=X0, Y0=Y0, X1=X1, Y1=Y1, data_type="double")
        self.grd_delta_n2 = Grid(NX=NX, NY=NY, X0=X0, Y0=Y0, X1=X1, Y1=Y1, data_type="double")
        self.grd_term1 = Grid(NX=NX, NY=NY, X0=X0, Y0=Y0, X1=X1, Y1=Y1, data_type="double")
        self.grd_M1 = Grid(NX=NX, NY=NY, X0=X0, Y0=Y0, X1=X1, Y1=Y1, data_type="double")
        self.grd_M2 = Grid(NX=NX, NY=NY, X0=X0, Y0=Y0, X1=X1, Y1=Y1, data_type="double")
        self.grd_sum = Grid(NX=NX, NY=NY, X0=X0, Y0=Y0, X1=X1, Y1=Y1, data_type="double")
        self.grd_sum.data = [[0.0 for _ in range(NY)] for _ in range(NX)]

    def push(self, grd_data, mask):
        """
        Updates statistics with new grid data where the mask is True.
        """
        if not self.grd_sum:
            raise ValueError("Grids must be initialized before pushing data.")

        self.n += 1
        for i in range(len(grd_data)):
            for j in range(len(grd_data[0])):
                if mask[i][j]:
                    self.grd_sum.data[i][j] += grd_data[i][j]

    def mean(self):
        """
        Computes the mean of the data.
        """
        if not self.grd_sum or self.n == 0:
            raise ValueError("Grids must be initialized and contain data.")
        return np.array(self.grd_sum.data) / self.n

    def sum(self):
        """
        Computes the sum of the data.
        """
        if not self.grd_sum:
            raise ValueError("Grids must be initialized and contain data.")
        return np.array(self.grd_sum.data)

    def variance(self):
        """
        Computes the variance of the data.
        """
        if not self.grd_sum or self.n <= 1:
            raise ValueError("Not enough data points to calculate variance.")
        mean_grid = self.mean()
        sum_of_squares = np.array([
            [
                (self.grd_sum.data[i][j] - mean_grid[i][j]) ** 2
                for j in range(len(self.grd_sum.data[0]))
            ]
            for i in range(len(self.grd_sum.data))
        ])
        return sum_of_squares / (self.n - 1)

    def std_deviation(self):
        """
        Computes the standard deviation of the data.
        """
        return np.sqrt(self.variance())

if __name__ == "__main__":
    # Re-test with the push method included
    running_stats = RunningStats()
    running_stats.initialize(NX=10, NY=10, X0=0.0, Y0=0.0, X1=100.0, Y1=100.0, nodata_value=-9999.0)

    # Mock data and mask for testing
    mock_data = [[i + j for j in range(10)] for i in range(10)]
    mock_mask = [[True] * 10 for _ in range(10)]
    running_stats.push(mock_data, mock_mask)  # Add data

    # Compute statistics
    mean = running_stats.mean()
    total_sum = running_stats.sum()
    variance = running_stats.variance()
    std_dev = running_stats.std_deviation()

    mean, total_sum, variance, std_dev  # Display results
