class SnowmeltOriginal:
    """
    Python translation of the Fortran module snowmelt__original.
    """
    def __init__(self):
        # Constants
        self.FREEZING_F = 32.0  # Freezing point in Fahrenheit
        self.FREEZING_C = 0.0   # Freezing point in Celsius
        self.MELT_INDEX = 1.5   # Melt factor
        self.DEGC_PER_DEGF = 0.5555555555555556  # Conversion factor (°C per °F)
        self.MM_PER_INCH = 25.4  # Conversion factor (mm per inch)

    def calculate_snowmelt(self, tmin, tmax, imperial_units=True):
        """
        Calculates potential snowmelt based on temperature and unit system.

        Parameters:
        - tmin (float): Minimum temperature.
        - tmax (float): Maximum temperature.
        - imperial_units (bool): True if temperatures are in Fahrenheit, False if in Celsius.

        Returns:
        - potential_snowmelt (float): The potential snowmelt.
        """
        potential_snowmelt = 0.0

        if imperial_units:
            # Imperial units: temperatures in °F, snowmelt in inches
            if tmax > self.FREEZING_F:
                melt_temp = max(0.0, (tmin + tmax) / 2 - self.FREEZING_F)
                potential_snowmelt = self.MELT_INDEX * melt_temp
        else:
            # Metric units: temperatures in °C, snowmelt in mm
            if tmax > self.FREEZING_C:
                melt_temp = max(0.0, (tmin + tmax) / 2 - self.FREEZING_C)
                potential_snowmelt = self.MELT_INDEX * melt_temp * self.MM_PER_INCH

        return potential_snowmelt

if __name__ == "__main__":
    # Example usage
    snowmelt_model = SnowmeltOriginal()

    # Test the calculation with example inputs
    tmin_f = 30.0  # Minimum temperature in Fahrenheit
    tmax_f = 40.0  # Maximum temperature in Fahrenheit

    tmin_c = -1.0  # Minimum temperature in Celsius
    tmax_c = 5.0   # Maximum temperature in Celsius

    # Calculate snowmelt in imperial and metric units
    snowmelt_imperial = snowmelt_model.calculate_snowmelt(tmin_f, tmax_f, imperial_units=True)
    snowmelt_metric = snowmelt_model.calculate_snowmelt(tmin_c, tmax_c, imperial_units=False)

    snowmelt_imperial, snowmelt_metric  # Display results

