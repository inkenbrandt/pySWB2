
import numpy as np

class InterceptionBucket:
    def __init__(self):
        self.iLanduseCodes = None
        self.INTERCEPTION_A_VALUE_GROWING_SEASON = None
        self.INTERCEPTION_B_VALUE_GROWING_SEASON = None
        self.INTERCEPTION_N_VALUE_GROWING_SEASON = None
        self.INTERCEPTION_A_VALUE_NONGROWING_SEASON = None
        self.INTERCEPTION_B_VALUE_NONGROWING_SEASON = None
        self.INTERCEPTION_N_VALUE_NONGROWING_SEASON = None

    def initialize(self, landuse_codes, a_values_gs, b_values_gs, n_values_gs, 
                   a_values_ngs, b_values_ngs, n_values_ngs):
        self.iLanduseCodes = np.array(landuse_codes, dtype=int)
        self.INTERCEPTION_A_VALUE_GROWING_SEASON = np.array(a_values_gs, dtype=float)
        self.INTERCEPTION_B_VALUE_GROWING_SEASON = np.array(b_values_gs, dtype=float)
        self.INTERCEPTION_N_VALUE_GROWING_SEASON = np.array(n_values_gs, dtype=float)
        self.INTERCEPTION_A_VALUE_NONGROWING_SEASON = np.array(a_values_ngs, dtype=float)
        self.INTERCEPTION_B_VALUE_NONGROWING_SEASON = np.array(b_values_ngs, dtype=float)
        self.INTERCEPTION_N_VALUE_NONGROWING_SEASON = np.array(n_values_ngs, dtype=float)

    def calculate(self, input_values, season="growing"):
        if season == "growing":
            return (
                self.INTERCEPTION_A_VALUE_GROWING_SEASON * input_values**2 +
                self.INTERCEPTION_B_VALUE_GROWING_SEASON * input_values +
                self.INTERCEPTION_N_VALUE_GROWING_SEASON
            )
        elif season == "nongrowing":
            return (
                self.INTERCEPTION_A_VALUE_NONGROWING_SEASON * input_values**2 +
                self.INTERCEPTION_B_VALUE_NONGROWING_SEASON * input_values +
                self.INTERCEPTION_N_VALUE_NONGROWING_SEASON
            )
        else:
            raise ValueError("Invalid season. Choose 'growing' or 'nongrowing'.")
