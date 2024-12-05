class RunoffCurveNumber:
    """
    Python translation of the Fortran module runoff__curve_number.
    """

    def __init__(self):
        # Allocated arrays
        self.CN_ARCIII = None  # Curve Number for ARC III (Antecedent Runoff Condition)
        self.CN_ARCII = None  # Curve Number for ARC II
        self.CN_ARCI = None  # Curve Number for ARC I
        self.PREV_5_DAYS_RAIN = None  # Previous 5 days' rainfall
        self.iLanduseCodes = None  # Land use codes

        # Scalar variables
        self.DAYCOUNT = 0  # Day counter
        self.FIVE_DAY_SUM = 6  # Constant parameter

    def initialize(self, shape, default_rainfall=0.0, default_CN=0.0):
        """
        Initializes the arrays and constants.

        Parameters:
        - shape (tuple): Shape of the 2D arrays (e.g., (rows, cols)).
        - default_rainfall (float): Default value for rainfall.
        - default_CN (float): Default curve number value.
        """
        self.CN_ARCIII = [[default_CN for _ in range(shape[1])] for _ in range(shape[0])]
        self.CN_ARCII = [[default_CN for _ in range(shape[1])] for _ in range(shape[0])]
        self.CN_ARCI = [[default_CN for _ in range(shape[1])] for _ in range(shape[0])]
        self.PREV_5_DAYS_RAIN = [[default_rainfall for _ in range(shape[1])] for _ in range(shape[0])]
        self.iLanduseCodes = [0] * shape[0]  # Example: 1D array for land use codes

    def update_previous_5_day_rainfall(self, new_rainfall):
        """
        Updates the previous 5 days' rainfall with new rainfall data.

        Parameters:
        - new_rainfall (list): A 2D list of rainfall values matching the array shape.
        """
        for i in range(len(self.PREV_5_DAYS_RAIN)):
            for j in range(len(self.PREV_5_DAYS_RAIN[0])):
                # Shift rainfall history: drop the oldest value and append the new one
                self.PREV_5_DAYS_RAIN[i][j] = max(
                    0.0, self.PREV_5_DAYS_RAIN[i][j] + new_rainfall[i][j]
                )

    def calculate_runoff(self, rainfall, CN, threshold=0.2):
        """
        Calculates runoff based on the SCS Curve Number method.

        Parameters:
        - rainfall (list): A 2D list of rainfall values.
        - CN (list): A 2D list of curve numbers.
        - threshold (float): The initial abstraction ratio (default: 0.2).

        Returns:
        - runoff (list): Calculated runoff for each grid cell.
        """
        runoff = []
        for i in range(len(rainfall)):
            row = []
            for j in range(len(rainfall[0])):
                S = (1000 / CN[i][j] - 10) if CN[i][j] != 0 else 0  # Potential max retention
                Ia = threshold * S  # Initial abstraction
                if rainfall[i][j] > Ia:
                    runoff_value = ((rainfall[i][j] - Ia) ** 2) / (rainfall[i][j] - Ia + S)
                else:
                    runoff_value = 0.0
                row.append(runoff_value)
            runoff.append(row)
        return runoff

if __name__ == "__main__":
    # Testing the functionality
    runoff_module = RunoffCurveNumber()
    runoff_module.initialize((5, 5), default_rainfall=0.5, default_CN=75)

    # Simulate new rainfall data
    new_rainfall_data = [[1.0] * 5 for _ in range(5)]
    runoff_module.update_previous_5_day_rainfall(new_rainfall_data)

    # Simulate runoff calculation
    rainfall_data = [[2.0] * 5 for _ in range(5)]
    curve_numbers = [[80] * 5 for _ in range(5)]  # Example curve numbers
    runoff_results = runoff_module.calculate_runoff(rainfall_data, curve_numbers)

    runoff_results  # Display the calculated runoff values
