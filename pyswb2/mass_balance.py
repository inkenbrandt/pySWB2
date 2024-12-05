class SoilMassBalance:
    """
    Python translation of the Fortran module mass_balance__soil.
    """
    def calculate_soil_mass_balance(
        self,
        net_infiltration,
        soil_storage,
        actual_et_soil,
        runoff,
        delta_soil_storage,
        reference_et0,
        soil_storage_max,
        infiltration,
    ):
        """
        Calculates soil mass balance.

        Parameters:
        - net_infiltration (float): Net infiltration into the soil.
        - soil_storage (float): Current soil storage.
        - actual_et_soil (float): Actual evapotranspiration from the soil.
        - runoff (float): Surface runoff.
        - delta_soil_storage (float): Change in soil storage.
        - reference_et0 (float): Reference evapotranspiration.
        - soil_storage_max (float): Maximum soil storage capacity.
        - infiltration (float): Infiltration capacity.

        Returns:
        - Updated values for soil storage, actual ET, runoff, and delta soil storage.
        """
        # Update soil storage with net infiltration
        soil_storage += net_infiltration

        # Calculate potential ET limited by soil storage
        potential_et = min(reference_et0, soil_storage)
        actual_et_soil = potential_et
        soil_storage -= actual_et_soil

        # Handle runoff if storage exceeds capacity
        if soil_storage > soil_storage_max:
            runoff = soil_storage - soil_storage_max
            soil_storage = soil_storage_max
        else:
            runoff = 0.0

        # Calculate change in soil storage
        delta_soil_storage = soil_storage - delta_soil_storage

        return {
            "soil_storage": soil_storage,
            "actual_et_soil": actual_et_soil,
            "runoff": runoff,
            "delta_soil_storage": delta_soil_storage,
        }


class ImperviousSurfaceMassBalance:
    """
    Python translation of the Fortran module mass_balance__impervious_surface.
    """
    NEAR_ZERO = 1.0e-6  # Threshold for near-zero values

    def calculate_impervious_surface_mass_balance(
        self,
        surface_storage,
        actual_et_impervious,
        paved_to_unpaved,
        surface_storage_max,
        storm_drain_capture,
        storm_drain_capture_fraction,
        net_rainfall,
        snowmelt,
    ):
        """
        Calculates the mass balance for impervious surfaces.

        Parameters:
        - surface_storage (float): Current surface storage on impervious surfaces.
        - actual_et_impervious (float): Evapotranspiration from impervious surfaces.
        - paved_to_unpaved (float): Water transfer from paved to unpaved surfaces.
        - surface_storage_max (float): Maximum storage capacity on impervious surfaces.
        - storm_drain_capture (float): Water captured by storm drains.
        - storm_drain_capture_fraction (float): Fraction of water captured by storm drains.
        - net_rainfall (float): Net rainfall on impervious surfaces.
        - snowmelt (float): Snowmelt contribution.

        Returns:
        - Updated values for surface storage, ET, storm drain capture, and paved-to-unpaved flow.
        """
        # Add net rainfall and snowmelt to surface storage
        surface_storage += net_rainfall + snowmelt

        # Calculate potential ET limited by surface storage
        actual_et_impervious = min(surface_storage, actual_et_impervious)
        surface_storage -= actual_et_impervious

        # Calculate storm drain capture based on the capture fraction
        storm_drain_capture = storm_drain_capture_fraction * surface_storage
        surface_storage -= storm_drain_capture

        # Calculate paved-to-unpaved flow for excess storage
        if surface_storage > surface_storage_max:
            paved_to_unpaved = surface_storage - surface_storage_max
            surface_storage = surface_storage_max
        else:
            paved_to_unpaved = 0.0

        return {
            "surface_storage": surface_storage,
            "actual_et_impervious": actual_et_impervious,
            "storm_drain_capture": storm_drain_capture,
            "paved_to_unpaved": paved_to_unpaved,
        }

class InterceptionMassBalance:
    """
    Python translation of the Fortran module mass_balance__interception.
    """
    def calculate_interception_mass_balance(
        self,
        interception_storage,
        actual_et_interception,
        interception,
        interception_storage_max,
        reference_et0,
    ):
        """
        Calculates the mass balance for interception storage.

        Parameters:
        - interception_storage (float): Current interception storage.
        - actual_et_interception (float): Evapotranspiration from interception.
        - interception (float): Additional interception from precipitation.
        - interception_storage_max (float): Maximum interception storage capacity.
        - reference_et0 (float): Reference evapotranspiration.

        Returns:
        - Updated values for interception storage and actual ET interception.
        """
        # Add interception to storage
        interception_storage += interception

        # Limit evapotranspiration by interception storage and reference ET
        actual_et_interception = min(interception_storage, reference_et0)
        interception_storage -= actual_et_interception

        # Ensure storage does not exceed maximum capacity
        if interception_storage > interception_storage_max:
            interception_storage = interception_storage_max

        return {
            "interception_storage": interception_storage,
            "actual_et_interception": actual_et_interception,
        }


class SnowMassBalance:
    """
    Python translation of the Fortran module mass_balance__snow.
    """
    def calculate_snow_mass_balance(self, snow_storage, potential_snowmelt, snowmelt, net_snowfall):
        """
        Calculates the mass balance for snow storage.

        Parameters:
        - snow_storage (float): Current snow storage.
        - potential_snowmelt (float): Maximum potential snowmelt.
        - snowmelt (float): Actual snowmelt.
        - net_snowfall (float): Net snowfall added to storage.

        Returns:
        - Updated values for snow storage and actual snowmelt.
        """
        # Update snow storage with net snowfall, ensuring no negative storage
        snow_storage = max(0.0, snow_storage + net_snowfall)

        # Calculate actual snowmelt based on potential snowmelt and storage
        if snow_storage > potential_snowmelt:
            snowmelt = potential_snowmelt
            snow_storage -= snowmelt
        else:
            snowmelt = snow_storage
            snow_storage = 0.0

        return {
            "snow_storage": snow_storage,
            "snowmelt": snowmelt,
        }





