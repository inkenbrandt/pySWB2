import math

class ETGriddedValues:
    """
    Python translation of the Fortran module et__gridded_values.
    """
    def __init__(self):
        # Placeholder for ET grid data
        self.ET_GRID = None

    def initialize(self, grid_shape, default_value=0.0):
        """
        Initializes the ET grid with default values.

        Parameters:
        - grid_shape (tuple): Shape of the 2D ET grid (rows, cols).
        - default_value (float): Default ET value for each grid cell.
        """
        self.ET_GRID = [[default_value for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]

    def calculate(self, daily_values, active_grid):
        """
        Calculates potential ET by substituting daily average ET values.

        Parameters:
        - daily_values (list of lists): A 2D list of daily ET values matching the grid.
        - active_grid (list of lists): A 2D boolean grid indicating active cells.

        Returns:
        - Updated ET grid with calculated values.
        """
        for i in range(len(self.ET_GRID)):
            for j in range(len(self.ET_GRID[0])):
                if active_grid[i][j]:
                    self.ET_GRID[i][j] = daily_values[i][j]
        return self.ET_GRID

class ETHargreavesSamani:
    """
    Python translation of the Fortran module et__hargreaves_samani with extraterrestrial radiation calculation.
    """
    def __init__(self):
        # Constants for the Hargreaves-Samani equation
        self.KRS = 0.0023  # Empirical coefficient (Hargreaves constant)
        self.GSC = 0.0820  # Solar constant (MJ/m²/min)

    @staticmethod
    def solar_declination(day_of_year):
        """
        Calculates the solar declination angle (radians).

        Parameters:
        - day_of_year (int): Day of the year (1 to 365).

        Returns:
        - delta (float): Solar declination (radians).
        """
        return 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)

    @staticmethod
    def inverse_relative_distance(day_of_year):
        """
        Calculates the inverse relative distance Earth-Sun.

        Parameters:
        - day_of_year (int): Day of the year (1 to 365).

        Returns:
        - dr (float): Inverse relative distance Earth-Sun.
        """
        return 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)

    @staticmethod
    def sunset_hour_angle(latitude, solar_declination):
        """
        Calculates the sunset hour angle (radians).

        Parameters:
        - latitude (float): Latitude in radians.
        - solar_declination (float): Solar declination angle (radians).

        Returns:
        - omega_s (float): Sunset hour angle (radians).
        """
        return math.acos(-math.tan(latitude) * math.tan(solar_declination))

    def extraterrestrial_radiation(self, latitude, day_of_year):
        """
        Calculates extraterrestrial radiation (Ra).

        Parameters:
        - latitude (float): Latitude in degrees.
        - day_of_year (int): Day of the year (1 to 365).

        Returns:
        - Ra (float): Extraterrestrial radiation (MJ/m²/day).
        """
        # Convert latitude to radians
        phi = math.radians(latitude)
        # Calculate solar parameters
        delta = self.solar_declination(day_of_year)
        dr = self.inverse_relative_distance(day_of_year)
        omega_s = self.sunset_hour_angle(phi, delta)
        # Calculate Ra
        Ra = (
            (24 * 60 / math.pi) * self.GSC * dr *
            (omega_s * math.sin(phi) * math.sin(delta) +
             math.cos(phi) * math.cos(delta) * math.sin(omega_s))
        )
        return Ra

    def calculate_et(self, tmin, tmax, tmean, latitude, day_of_year):
        """
        Calculates potential ET using the Hargreaves-Samani method.

        Parameters:
        - tmin (float): Minimum temperature (°C).
        - tmax (float): Maximum temperature (°C).
        - tmean (float): Mean temperature (°C).
        - latitude (float): Latitude in degrees.
        - day_of_year (int): Day of the year (1 to 365).

        Returns:
        - ET_o (float): Reference evapotranspiration (mm/day).
        """
        # Calculate extraterrestrial radiation
        Ra = self.extraterrestrial_radiation(latitude, day_of_year)

        # Apply Hargreaves-Samani equation
        delta_t = tmax - tmin  # Temperature difference
        et_o = self.KRS * Ra * (tmean + 17.8) * (delta_t ** 0.5)

        return et_o

class ETJensenHaise:
    """
    Python translation of the Fortran module et__jensen_haise.
    """
    def __init__(self, base_temperature=0.0, coefficient=0.025):
        """
        Initializes the Jensen-Haise ET calculation parameters.

        Parameters:
        - base_temperature (float): Base temperature (\( T_b \)) in °C. Default is 0.0.
        - coefficient (float): Empirical coefficient (\( C \)) in mm/MJ. Default is 0.025.
        """
        self.T_b = base_temperature  # Base temperature (°C)
        self.C = coefficient  # Empirical coefficient (mm/MJ)

    def calculate_et(self, tmean, solar_radiation):
        """
        Calculates potential ET using the Jensen-Haise method.

        Parameters:
        - tmean (float): Mean temperature (\( T_{mean} \)) in °C.
        - solar_radiation (float): Solar radiation (\( R_s \)) in MJ/m²/day.

        Returns:
        - ET_o (float): Reference evapotranspiration (mm/day).
        """
        if tmean > self.T_b:
            return self.C * solar_radiation * (tmean - self.T_b)
        else:
            return 0.0

class ETZoneValues:
    """
    Python translation of the Fortran module et__zone_values.
    """
    def __init__(self):
        # Placeholder for ET table values and ratios
        self.ET_TABLE_VALUES = None  # 2D table of ET values by zone and time
        self.ET_ZONE = None  # Array of zone identifiers
        self.ET_RATIOS = None  # Ratios for ET adjustment

    def initialize(self, num_zones, num_timesteps, default_value=0.0, default_ratio=1.0):
        """
        Initializes the ET table, zone identifiers, and ratios.

        Parameters:
        - num_zones (int): Number of zones.
        - num_timesteps (int): Number of timesteps in the ET table.
        - default_value (float): Default ET value for each zone and timestep.
        - default_ratio (float): Default ET ratio for each zone.
        """
        self.ET_TABLE_VALUES = [[default_value for _ in range(num_timesteps)] for _ in range(num_zones)]
        self.ET_ZONE = [0] * num_zones  # Initialize zone identifiers to 0
        self.ET_RATIOS = [default_ratio] * num_zones

    def calculate(self, zone_ids, timestep):
        """
        Calculates ET values for a given timestep based on zone IDs and ET ratios.

        Parameters:
        - zone_ids (list of int): List of zone identifiers for each grid cell.
        - timestep (int): The current timestep.

        Returns:
        - et_values (list): List of ET values for each grid cell.
        """
        if timestep >= len(self.ET_TABLE_VALUES[0]):
            raise ValueError("Timestep out of range.")

        et_values = []
        for zone_id in zone_ids:
            if zone_id < 0 or zone_id >= len(self.ET_TABLE_VALUES):
                et_value = 0.0  # Default ET value for invalid zone IDs
            else:
                et_value = self.ET_TABLE_VALUES[zone_id][timestep] * self.ET_RATIOS[zone_id]
            et_values.append(et_value)

        return et_values

