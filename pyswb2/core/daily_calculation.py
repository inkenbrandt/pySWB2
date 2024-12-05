from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import logging

# Import from previously created modules
from model_domain import ModelDomain


class DailyCalculation:
    """Class handling daily water balance calculations"""
    
    def __init__(self, domain: ModelDomain):
        self.domain = domain
        self.logger = logging.getLogger('daily_calculation')
        
    def perform_daily_calculation(self):
        """Perform daily water balance calculations"""
        # Calculate degree days and update crop coefficients
        self.domain.calc_gdd()
        self.domain.update_crop_coefficient()
        self.domain.update_growing_season()
        self.domain.update_rooting_depth()
        self.domain.calc_reference_et()
        
        # Calculate fog and interception
        self.domain.calc_fog()
        self.domain.calc_interception()
        
        # Update interception storage and ET
        self._calculate_interception_mass_balance()
        
        # Update crop ET
        self.domain.crop_etc = np.maximum(
            self.domain.reference_et0 - self.domain.actual_et_interception,
            0.0
        ) * self.domain.crop_coefficient_kcb
        
        # Snow calculations
        self.domain.calc_snowfall()
        self.domain.calc_snowmelt()
        self.domain.calc_continuous_frozen_ground_index()
        self._calculate_snow_mass_balance()
        
        # Initialize runoff components
        self.domain.runon = np.zeros_like(self.domain.runon)
        self.domain.runoff = np.zeros_like(self.domain.runoff)
        self.domain.runoff_outside = np.zeros_like(self.domain.runoff_outside)
        
        # Process cells in order
        for jndx in range(len(self.domain.sort_order)):
            indx = self.domain.sort_order[jndx]
            self._process_cell(indx)
    
    def _calculate_interception_mass_balance(self):
        """Calculate interception storage mass balance"""
        temp_storage = self.domain.interception_storage + self.domain.interception
        
        mask = temp_storage > self.domain.interception_storage_max
        self.domain.interception[mask] = (
            self.domain.interception_storage_max[mask] - 
            self.domain.interception_storage[mask]
        )
        self.domain.interception_storage[mask] = self.domain.interception_storage_max[mask]
        
        self.domain.actual_et_interception = np.minimum(
            self.domain.reference_et0,
            self.domain.interception_storage
        )
        self.domain.interception_storage = np.maximum(
            0.0,
            self.domain.interception_storage - self.domain.actual_et_interception
        )
    
    def _calculate_snow_mass_balance(self):
        """Calculate snow storage mass balance"""
        self.domain.snow_storage = np.maximum(
            0.0,
            self.domain.snow_storage + self.domain.net_snowfall
        )
        
        mask = self.domain.snow_storage > self.domain.potential_snowmelt
        self.domain.snowmelt[mask] = self.domain.potential_snowmelt[mask]
        self.domain.snow_storage[mask] -= self.domain.snowmelt[mask]
        
        mask = ~mask
        self.domain.snowmelt[mask] = self.domain.snow_storage[mask]
        self.domain.snow_storage[mask] = 0.0
    
    def _calculate_impervious_surface_mass_balance(self, indx: int):
        """Calculate impervious surface mass balance for a cell"""
        cell = self._get_cell_data(indx)
        
        # Update surface storage
        cell.surface_storage += (
            cell.net_rainfall +
            cell.fog +
            cell.snowmelt -
            cell.runoff
        )
        
        # Calculate excess storage
        surface_storage_excess = np.maximum(
            0.0,
            (cell.surface_storage - cell.surface_storage_max) *
            (1.0 - cell.pervious_fraction)
        )
        
        cell.surface_storage = np.minimum(
            cell.surface_storage,
            cell.surface_storage_max
        )
        
        # Calculate storm drain capture
        cell.storm_drain_capture = (
            surface_storage_excess *
            cell.storm_drain_capture_fraction
        )
        
        # Calculate paved to unpaved transfer
        cell.paved_to_unpaved = np.maximum(
            0.0,
            surface_storage_excess - cell.storm_drain_capture
        )
        
        # Update ET
        cell.actual_et_impervious = np.minimum(
            cell.reference_et0,
            cell.surface_storage
        )
        cell.surface_storage = np.maximum(
            cell.surface_storage - cell.actual_et_impervious,
            0.0
        )
    
    def _calculate_soil_mass_balance(self, indx: int):
        """Calculate soil moisture mass balance for a cell"""
        cell = self._get_cell_data(indx)
        
        new_soil_storage = (
            cell.soil_storage +
            cell.infiltration -
            cell.actual_et_soil
        )
        
        # Handle special cases
        if cell.soil_storage_max < 1.0e-6:
            # Open water cell
            cell.actual_et_soil = np.minimum(
                cell.reference_et0,
                cell.infiltration
            )
            cell.net_infiltration = 0.0
            cell.runoff = np.maximum(
                0.0,
                cell.infiltration + cell.runoff - cell.actual_et_soil
            )
            cell.soil_storage = 0.0
            cell.delta_soil_storage = 0.0
        
        elif new_soil_storage > cell.soil_storage_max:
            # Excess soil moisture
            cell.net_infiltration = new_soil_storage - cell.soil_storage_max
            cell.delta_soil_storage = cell.soil_storage_max - cell.soil_storage
            cell.soil_storage = cell.soil_storage_max
        
        else:
            # Normal case
            cell.delta_soil_storage = new_soil_storage - cell.soil_storage
            cell.soil_storage = new_soil_storage
            cell.net_infiltration = 0.0
    
    def _get_cell_data(self, indx: int):
        """Helper method to get cell data as a simple object"""
        @dataclass
        class CellData:
            surface_storage: float
            actual_et_impervious: float 
            paved_to_unpaved: float
            surface_storage_max: float
            storm_drain_capture: float
            storm_drain_capture_fraction: float
            net_rainfall: float
            snowmelt: float
            runon: float
            runoff: float
            fog: float
            reference_et0: float
            pervious_fraction: float
            soil_storage: float
            actual_et_soil: float
            delta_soil_storage: float
            soil_storage_max: float
            infiltration: float
            net_infiltration: float
        
        return CellData(
            surface_storage=self.domain.surface_storage[indx],
            actual_et_impervious=self.domain.actual_et_impervious[indx],
            paved_to_unpaved=self.domain.surface_storage_excess[indx],
            surface_storage_max=self.domain.surface_storage_max[indx],
            storm_drain_capture=self.domain.storm_drain_capture[indx],
            storm_drain_capture_fraction=self.domain.storm_drain_capture_fraction[indx],
            net_rainfall=self.domain.net_rainfall[indx],
            snowmelt=self.domain.snowmelt[indx],
            runon=self.domain.runon[indx],
            runoff=self.domain.runoff[indx],
            fog=self.domain.fog[indx],
            reference_et0=self.domain.reference_et0[indx],
            pervious_fraction=self.domain.pervious_fraction[indx],
            soil_storage=self.domain.soil_storage[indx],
            actual_et_soil=self.domain.actual_et_soil[indx],
            delta_soil_storage=self.domain.delta_soil_storage[indx],
            soil_storage_max=self.domain.soil_storage_max[indx],
            infiltration=self.domain.infiltration[indx],
            net_infiltration=self.domain.net_infiltration[indx]
        )

    def _process_cell(self, indx: int):
        """Process calculations for a single cell"""
        # Calculate inflow
        inflow = np.maximum(
            0.0,
            self.domain.runon[indx] +
            self.domain.net_rainfall[indx] +
            self.domain.fog[indx] +
            self.domain.snowmelt[indx]
        )
        
        # Calculate runoff and irrigation
        self.domain.calc_runoff(indx)
        self.domain.calc_irrigation(indx)
        
        # Mass balance calculations
        self._calculate_impervious_surface_mass_balance(indx)
        self.domain.calc_direct_soil_moisture(indx)
        
        # Calculate infiltration
        self.domain.infiltration[indx] = np.maximum(
            0.0,
            ((self.domain.runon[indx] +
              self.domain.net_rainfall[indx] +
              self.domain.fog[indx] +
              self.domain.snowmelt[indx] +
              self.domain.irrigation[indx] +
              self.domain.direct_soil_moisture[indx] -
              self.domain.runoff[indx]) *
             self.domain.pervious_fraction[indx] +
             self.domain.surface_storage_excess[indx]) /
            self.domain.pervious_fraction[indx]
        )
        
        # Calculate actual ET
        self.domain.calc_actual_et(indx)
        
        # Soil mass balance
        self._calculate_soil_mass_balance(indx)
        
        # Update actual ET for entire cell
        self.domain.actual_et[indx] = (
            self.domain.actual_et_soil[indx] *
            self.domain.pervious_fraction[indx] +
            self.domain.actual_et_impervious[indx] *
            (1.0 - self.domain.pervious_fraction[indx]) +
            self.domain.actual_et_interception[indx] *
            self.domain.canopy_cover_fraction[indx]
        )
        
        # Handle negative runoff
        if self.domain.runoff[indx] < 0:
            self.logger.warning(
                f"Negative runoff at index {indx}, "
                f"col={self.domain.col_num_1D[indx]}, "
                f"row={self.domain.row_num_1D[indx]}"
            )
        
        # Final calculations
        self.domain.calc_direct_net_infiltration(indx)
        self.domain.net_infiltration[indx] = (
            self.domain.net_infiltration[indx] *
            self.domain.pervious_fraction[indx] +
            self.domain.direct_net_infiltration[indx]
        )
        self.domain.irrigation[indx] *= self.domain.pervious_fraction[indx]
        
        # Maximum infiltration and routing
        self.domain.calc_maximum_net_infiltration(indx)
        self.domain.calc_routing(indx)
