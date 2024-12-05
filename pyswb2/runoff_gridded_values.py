from datetime import datetime


class RunoffGriddedValues:
    """
    Python translation of the Fortran module runoff__gridded_values.
    """

    def __init__(self):
        # Allocated arrays
        self.RUNOFF_TABLE_VALUES = None  # 2D table of runoff values
        self.RUNOFF_TABLE_DATES = None  # Dates corresponding to the table values
        self.pRUNOFF_ZONE = None  # Placeholder for runoff zone data
        self.RUNOFF_ZONE = None  # Array of runoff zone data
        self.RUNOFF_RATIOS = None  # Array of runoff ratios

    def initialize(self, shape_values, shape_zones, default_value=0.0, default_ratio=1.0):
        """
        Initializes the arrays and default values.

        Parameters:
        - shape_values (tuple): Shape of the 2D runoff table values array.
        - shape_zones (int): Size of the runoff zone array.
        - default_value (float): Default value for runoff table values.
        - default_ratio (float): Default value for runoff ratios.
        """
        self.RUNOFF_TABLE_VALUES = [[default_value for _ in range(shape_values[1])] for _ in range(shape_values[0])]
        self.RUNOFF_TABLE_DATES = [None] * shape_values[0]  # Placeholder for dates
        self.RUNOFF_ZONE = [0] * shape_zones
        self.RUNOFF_RATIOS = [default_ratio] * shape_zones

    def update_ratios(self, zone_factors):
        """
        Updates the runoff ratios based on zone-specific factors.

        Parameters:
        - zone_factors (list): List of factors to apply to each zone.
        """
        if len(zone_factors) != len(self.RUNOFF_ZONE):
            raise ValueError("Zone factors must match the number of zones.")

        self.RUNOFF_RATIOS = [
            self.RUNOFF_RATIOS[i] * zone_factors[i] for i in range(len(self.RUNOFF_ZONE))
        ]

    def update_table_by_date(self, date, new_values):
        """
        Updates the runoff table for a specific date.

        Parameters:
        - date (datetime): The date to update.
        - new_values (list): A list of new runoff values for the given date.
        """
        if len(new_values) != len(self.RUNOFF_TABLE_VALUES[0]):
            raise ValueError("New values must match the number of columns in the table.")

        for i, existing_date in enumerate(self.RUNOFF_TABLE_DATES):
            if existing_date == date:
                self.RUNOFF_TABLE_VALUES[i] = new_values
                return

        # If the date does not exist, add a new row
        self.RUNOFF_TABLE_DATES.append(date)
        self.RUNOFF_TABLE_VALUES.append(new_values)

if __name__ == "__main__":
    # Example usage
    runoff_values = RunoffGriddedValues()
    runoff_values.initialize(shape_values=(12, 5), shape_zones=5, default_value=0.5, default_ratio=1.0)

    # Update table with a specific date
    date_to_update = datetime(2024, 1, 1)
    new_values = [0.7, 0.8, 0.6, 0.9, 0.75]
    runoff_values.update_table_by_date(date_to_update, new_values)

    # Display updated table and dates
    runoff_values.RUNOFF_TABLE_VALUES, runoff_values.RUNOFF_TABLE_DATES
