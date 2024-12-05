
import numpy as np

class InterceptionGash:
    def __init__(self):
        self.CANOPY_STORAGE_CAPACITY_TABLE_VALUES = None
        self.TRUNK_STORAGE_CAPACITY_TABLE_VALUES = None
        self.STEMFLOW_FRACTION_TABLE_VALUES = None
        self.EVAPORATION_TO_RAINFALL_RATIO = None
        self.P_SAT = None

    def initialize(self, canopy_values, trunk_values, stemflow_values, evap_to_rain_ratio, p_sat):
        self.CANOPY_STORAGE_CAPACITY_TABLE_VALUES = np.array(canopy_values, dtype=float)
        self.TRUNK_STORAGE_CAPACITY_TABLE_VALUES = np.array(trunk_values, dtype=float)
        self.STEMFLOW_FRACTION_TABLE_VALUES = np.array(stemflow_values, dtype=float)
        self.EVAPORATION_TO_RAINFALL_RATIO = float(evap_to_rain_ratio)
        self.P_SAT = float(p_sat)

    def calculate(self, precipitation):
        # Simple calculation for demonstration purposes
        intercepted_precipitation = np.minimum(
            precipitation, self.CANOPY_STORAGE_CAPACITY_TABLE_VALUES
        )
        evaporation = intercepted_precipitation * self.EVAPORATION_TO_RAINFALL_RATIO
        return intercepted_precipitation - evaporation

    def precipitation_at_saturation(self):
        return self.P_SAT
