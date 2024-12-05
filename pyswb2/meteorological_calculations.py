import math

class MeteorologicalCalculations:
    """
    Python translation of the Fortran module meteorological_calculations.
    """
    @staticmethod
    def sat_vapor_pressure(temperature):
        """
        Calculates saturation vapor pressure (e_0) in kilopascals.

        Parameters:
        - temperature (float): Air temperature in °C.

        Returns:
        - e_0 (float): Saturation vapor pressure in kilopascals.
        """
        return 0.6108 * math.exp((17.27 * temperature) / (temperature + 237.3))

    @staticmethod
    def dewpoint_vapor_pressure(tmin):
        """
        Estimates dewpoint vapor pressure (e_a) in kilopascals.

        Parameters:
        - tmin (float): Minimum daily air temperature in °C.

        Returns:
        - e_a (float): Dewpoint vapor pressure in kilopascals.
        """
        return 0.6108 * math.exp((17.27 * tmin) / (tmin + 237.3))

    @staticmethod
    def equivalent_evaporation(radiation):
        """
        Converts radiation (MJ/m²/day) into equivalent evaporation (mm/day).

        Parameters:
        - radiation (float): Radiation in MJ/m²/day.

        Returns:
        - R_ET (float): Equivalent evaporation in mm/day.
        """
        return radiation * 0.408


