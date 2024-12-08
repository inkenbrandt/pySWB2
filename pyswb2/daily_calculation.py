from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
import logging


class DailyCalculation:
    """Class handling daily water balance calculations following documented order"""

    def __init__(self, domain):
        """Initialize daily calculation module

        Args:
            domain: ModelDomain instance containing state variables and modules
        """
        self.domain = domain
        self.logger = logging.getLogger('daily_calculation')

    def perform_daily_calculation(self, date: datetime) -> None:
        """Execute daily water balance calculations in proper order

        Args:
            date: Current simulation date
        """
        # 1. Partition precipitation into rain and snow
        self._partition_precipitation()

        # 2. Handle canopy interception
        self._process_interception()

        # 3. Update snow storage and calculate snowmelt
        self._process_snow()

        # 4. Calculate reference ET and crop ET
        self.domain.calc_reference_et()
        self.domain.crop_etc = self.domain.reference_et0 * self.domain.crop_coefficient

        # 5. Calculate soil moisture balance
        self._process_soil_moisture()

        # 6. Calculate water routing if enabled
        if self.domain.routing_enabled:
            self._process_routing()

        # 7. Update crop and growing season variables for next day
        self._update_growing_conditions()

    def _partition_precipitation(self) -> None:
        """Partition daily precipitation into rain and snow"""
        # Use mean temperature and daily range to determine precipitation type
        tmean = (self.domain.tmax + self.domain.tmin) / 2.0
        daily_range = (self.domain.tmax - self.domain.tmin) / 3.0

        snow_mask = (tmean - daily_range) <= 32.0

        # Partition precipitation based on temperature
        self.domain.snowfall[snow_mask] = self.domain.gross_precip[snow_mask]
        self.domain.rainfall[~snow_mask] = self.domain.gross_precip[~snow_mask]

        # Track net amounts after interception
        self.domain.net_snowfall = self.domain.snowfall.copy()
        self.domain.net_rainfall = self.domain.rainfall.copy()

    def _process_interception(self) -> None:
        """Calculate interception and update storages"""
        # Calculate new interception
        self.domain.interception_module.calculate_interception(
            rainfall=self.domain.rainfall,
            fog=self.domain.fog,
            is_growing_season=self.domain.agriculture_module.is_growing_season
        )

        # Update interception storage
        self._calculate_interception_mass_balance()

        # Reduce net precipitation by interception amounts
        self.domain.net_rainfall = np.maximum(
            0.0,
            self.domain.rainfall - self.domain.interception
        )
        self.domain.net_snowfall = np.maximum(
            0.0,
            self.domain.snowfall - self.domain.interception
        )

    def _calculate_interception_mass_balance(self) -> None:
        """Update interception storage mass balance"""
        # Add new interception to storage
        temp_storage = (
                self.domain.interception_storage +
                self.domain.interception
        )

        # Handle excess storage
        mask = temp_storage > self.domain.interception_storage_max
        self.domain.interception[mask] = (
                self.domain.interception_storage_max[mask] -
                self.domain.interception_storage[mask]
        )
        self.domain.interception_storage[mask] = self.domain.interception_storage_max[mask]

        # Calculate ET from interception
        self.domain.actual_et_interception = np.minimum(
            self.domain.reference_et0,
            self.domain.interception_storage
        )

        # Update final storage
        self.domain.interception_storage = np.maximum(
            0.0,
            self.domain.interception_storage - self.domain.actual_et_interception
        )

    def _process_snow(self) -> None:
        """Process snow accumulation and melt"""
        # Add new snow to storage
        self.domain.snow_storage = np.maximum(
            0.0,
            self.domain.snow_storage + self.domain.net_snowfall
        )

        # Calculate potential snowmelt
        self.domain.potential_snowmelt = np.where(
            self.domain.tmax > 32.0,
            0.059 * (self.domain.tmax - 32.0),  # 1.5mm per degree C above freezing
            0.0
        )

        # Calculate actual snowmelt based on available snow
        mask = self.domain.snow_storage > self.domain.potential_snowmelt
        self.domain.snowmelt[mask] = self.domain.potential_snowmelt[mask]
        self.domain.snow_storage[mask] -= self.domain.snowmelt[mask]

        # If less snow than potential melt, melt all remaining snow
        mask = ~mask
        self.domain.snowmelt[mask] = self.domain.snow_storage[mask]
        self.domain.snow_storage[mask] = 0.0

    def _process_soil_moisture(self) -> None:
        """Calculate soil moisture balance and net infiltration"""
        # Calculate interim soil moisture before ET
        interim_moisture = (
                self.domain.soil_storage +
                self.domain.net_rainfall +
                self.domain.snowmelt +
                self.domain.runon +
                self.domain.direct_soil_moisture -
                self.domain.runoff
        )

        # Calculate soil moisture fraction
        moisture_fraction = np.clip(
            (interim_moisture - self.domain.wilting_point) /
            (self.domain.field_capacity - self.domain.wilting_point),
            0.0, 1.0
        )

        # Calculate actual ET
        self.domain.actual_et = self.domain.actual_et_calculator.calculate(
            soil_storage=interim_moisture,
            soil_storage_max=self.domain.soil_storage_max,
            infiltration=self.domain.infiltration,
            crop_etc=self.domain.crop_etc,
            reference_et0=self.domain.reference_et0,
            **{
                'kcb': self.domain.crop_coefficient,
                'landuse_indices': self.domain.landuse_indices,
                'soil_group': self.domain.soil_indices,
                'awc': self.domain.available_water_content,
                'current_rooting_depth': self.domain.rooting_depth,
                'is_growing_season': self.domain.agriculture_module.is_growing_season
            }
        )

        # Update soil moisture
        self.domain.soil_storage = np.maximum(
            0.0,
            interim_moisture - self.domain.actual_et
        )

        # Calculate net infiltration
        self.domain.net_infiltration = np.where(
            self.domain.soil_storage > self.domain.field_capacity,
            self.domain.soil_storage - self.domain.field_capacity,
            0.0
        )

        # Limit soil storage to field capacity
        self.domain.soil_storage = np.minimum(
            self.domain.soil_storage,
            self.domain.field_capacity
        )

        # Add direct net infiltration
        self.domain.net_infiltration += self.domain.direct_net_infiltration

    def _process_routing(self) -> None:
        """Route surface water based on flow direction"""
        # Initialize routing arrays
        self.domain.runoff_outside = np.zeros_like(self.domain.runoff)
        self.domain.runon = np.zeros_like(self.domain.runoff)

        # Process cells in flow sequence order
        for idx in self.domain.flow_sequence:
            # Get downslope cell index
            next_idx = self.domain.flow_direction[idx]

            # Check if there's a valid downslope cell
            if next_idx == -1:
                self.domain.runoff_outside[idx] = self.domain.runoff[idx]
                continue

            # Route water to next cell
            self.domain.runon[next_idx] += (
                    self.domain.runoff[idx] *
                    self.domain.routing_fraction[idx]
            )

            # Track water leaving domain
            self.domain.runoff_outside[idx] = (
                    self.domain.runoff[idx] *
                    (1.0 - self.domain.routing_fraction[idx])
            )

    def _update_growing_conditions(self) -> None:
        """Update growing season and crop parameters"""
        # Update growing degree days
        self.domain.agriculture_module.calculate_growing_degree_days(
            self.domain.tmean,
            self.domain.tmin,
            self.domain.tmax
        )

        # Update growing season status
        self.domain.agriculture_module.update_growing_season(
            self.domain.tmean
        )

        # Update rooting depth if using dynamic rooting
        if self.domain.dynamic_rooting:
            self.domain.agriculture_module.update_root_depth()

        # Update crop coefficients if using FAO-56
        if self.domain.use_crop_coefficients:
            self.domain.agriculture_module.update_crop_coefficients()