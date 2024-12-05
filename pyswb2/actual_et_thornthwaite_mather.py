
import numpy as np

class ThornthwaiteMatherET:
    """Class for calculating actual evapotranspiration (ET) using the Thornthwaite-Mather method."""

    NEAR_ZERO = 1.0e-9  # Small value to handle numerical precision

    @staticmethod
    def calculate_actual_et(soil_storage, soil_storage_max, infiltration, potential_et):
        """Calculate actual ET using the Thornthwaite-Mather method.

        Args:
            soil_storage (array-like): Current soil storage.
            soil_storage_max (float): Maximum soil storage capacity.
            infiltration (array-like): Infiltration values.
            potential_et (array-like): Potential ET values.

        Returns:
            tuple: Updated actual ET and soil storage.
        """
        soil_storage = np.array(soil_storage, dtype=np.float32)
        infiltration = np.array(infiltration, dtype=np.float32)
        potential_et = np.array(potential_et, dtype=np.float32)
        
        actual_et = np.zeros_like(soil_storage, dtype=np.float32)

        # Calculate actual ET based on Thornthwaite-Mather rules
        for i in range(soil_storage.size):
            if potential_et[i] <= (soil_storage[i] + infiltration[i]):
                actual_et[i] = potential_et[i]
                soil_storage[i] += infiltration[i] - potential_et[i]
            else:
                actual_et[i] = soil_storage[i] + infiltration[i]
                soil_storage[i] = 0.0

            # Limit soil storage to maximum capacity
            if soil_storage[i] > soil_storage_max:
                soil_storage[i] = soil_storage_max

            # Handle precision issues with small numbers
            if soil_storage[i] < ThornthwaiteMatherET.NEAR_ZERO:
                soil_storage[i] = 0.0

        return actual_et, soil_storage
