
import numpy as np
from datetime import datetime

class StormDrainCapture:
    def __init__(self):
        self.storm_drain_capture_fraction = None
        self.storm_drain_capture_fraction_table = None
        self.date_of_last_retrieval = None

    def initialize(self, fraction_table, initial_date=None):
        """Initialize the storm drain capture module."""
        self.storm_drain_capture_fraction_table = np.array(fraction_table, dtype=float)
        self.storm_drain_capture_fraction = np.zeros(len(fraction_table), dtype=float)
        self.date_of_last_retrieval = initial_date if initial_date else datetime.now()

    def calculate(self, input_values):
        """Calculate storm drain capture fractions based on input values."""
        if self.storm_drain_capture_fraction_table is None:
            raise ValueError("The module has not been initialized with a fraction table.")

        # Example calculation: Use fraction_table to scale input_values
        self.storm_drain_capture_fraction = (
            input_values * self.storm_drain_capture_fraction_table
        )
        self.date_of_last_retrieval = datetime.now()

        return self.storm_drain_capture_fraction


# Example Usage
if __name__ == "__main__":
    # Example fraction table and input values
    fraction_table = [0.1, 0.2, 0.3, 0.4]
    input_values = np.array([10, 20, 30, 40], dtype=float)
    
    storm_drain = StormDrainCapture()
    storm_drain.initialize(fraction_table)
    results = storm_drain.calculate(input_values)

    print("Capture Fractions:", results)
    print("Last Retrieval Date:", storm_drain.date_of_last_retrieval)
