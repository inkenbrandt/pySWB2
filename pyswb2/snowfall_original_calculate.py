class SnowfallOriginal:
    """
    Python translation of the Fortran module snowfall__original.
    """
    def __init__(self):
        self.FREEZING = 32.0  # Freezing point in Fahrenheit

    def calculate_snowfall(
        self, tmin, tmax, interception, gross_precipitation
    ):
        """
        Calculates snowfall, rainfall, and net values based on temperature and precipitation.

        Parameters:
        - tmin (float): Minimum temperature.
        - tmax (float): Maximum temperature.
        - interception (float): Interception value.
        - gross_precipitation (float): Total precipitation.

        Returns:
        - snowfall (float): Snowfall amount.
        - net_snowfall (float): Snowfall after interception.
        - rainfall (float): Rainfall amount.
        - net_rainfall (float): Rainfall after interception.
        """
        # Classify gross precipitation into snowfall and rainfall based on temperature
        if tmax <= self.FREEZING:
            snowfall = gross_precipitation
            rainfall = 0.0
        elif tmin >= self.FREEZING:
            snowfall = 0.0
            rainfall = gross_precipitation
        else:
            fraction_snowfall = (self.FREEZING - tmin) / (tmax - tmin)
            snowfall = gross_precipitation * fraction_snowfall
            rainfall = gross_precipitation - snowfall

        # Calculate net values after interception
        net_snowfall = max(0.0, snowfall - interception)
        net_rainfall = max(0.0, rainfall - interception)

        return snowfall, net_snowfall, rainfall, net_rainfall

if __name__ == "__main__":
    # Example usage
    snowfall_model = SnowfallOriginal()

    # Test the calculation with example inputs
    tmin = 30.0  # Minimum temperature (Fahrenheit)
    tmax = 35.0  # Maximum temperature (Fahrenheit)
    interception = 0.5  # Interception value (inches)
    gross_precipitation = 2.0  # Total precipitation (inches)

    results = snowfall_model.calculate_snowfall(tmin, tmax, interception, gross_precipitation)
    results  # Display snowfall, net snowfall, rainfall, and net rainfall
