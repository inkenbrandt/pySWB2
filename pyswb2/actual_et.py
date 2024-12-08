from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
import numpy as np
from numpy.typing import NDArray
import logging


@dataclass
class FAO56Parameters:
    """Parameters for FAO-56 ET calculations"""
    depletion_fraction: float = 0.5  # Soil moisture depletion fraction
    rew: float = 0.0  # Readily evaporable water (inches)
    tew: float = 0.0  # Total evaporable water (inches)
    mean_plant_height: float = 0.0  # Mean plant height (feet)
    min_fraction_covered_soil: float = 0.05  # Minimum fraction of soil covered by vegetation

    # Two-stage method specific parameters
    kcb_min: float = 0.15  # Minimum basal crop coefficient
    kcb_mid: float = 1.10  # Mid-season basal crop coefficient
    kcb_end: float = 0.15  # End-season basal crop coefficient
    initial_root_depth: float = 50.0  # Initial rooting depth (mm)
    max_root_depth: float = 1500.0  # Maximum rooting depth (mm)
    root_growth_rate: float = 2.5  # Root growth rate (mm/day)


class ActualETCalculator:
    """Module for calculating actual evapotranspiration using different methods"""

    def __init__(self, domain_size: int,
                 method: Literal["fao56", "fao56_two_stage", "thornthwaite"] = "fao56_two_stage"):
        """Initialize ET calculator

        Args:
            domain_size: Number of cells in model domain
            method: ET calculation method to use
        """
        if method not in ["fao56", "fao56_two_stage", "thornthwaite"]:
            raise ValueError("method must be one of: 'fao56', 'fao56_two_stage', 'thornthwaite'")

        self.method = method
        self.domain_size = domain_size
        self.logger = logging.getLogger('actual_et')

        # Initialize arrays
        self.actual_et = np.zeros(domain_size, dtype=np.float32)
        self.depletion_fraction = np.full(domain_size, 0.5, dtype=np.float32)

        # Initialize indices
        self.landuse_indices: Optional[NDArray[np.int32]] = None
        self.soil_indices: Optional[NDArray[np.int32]] = None
        self.elevation: Optional[NDArray[np.float32]] = None

        # Add default parameters
        self.default_params = FAO56Parameters(
            depletion_fraction=0.5,
            rew=0.1,  # Default readily evaporable water (inches)
            tew=0.4,  # Default total evaporable water (inches)
            mean_plant_height=1.0,  # Default plant height (feet)
            min_fraction_covered_soil=0.05,
            kcb_min=0.15,
            kcb_mid=1.0,
            kcb_end=0.15,
            initial_root_depth=50.0,
            max_root_depth=1500.0,
            root_growth_rate=2.5
        )

        # Initialize parameters dictionary with default
        self.params = {0: self.default_params}  # Use 0 as default landuse code

        # Additional arrays for FAO56 two-stage method
        if method == "fao56_two_stage":
            self.evaporable_water_storage = np.zeros(domain_size, dtype=np.float32)
            self.evaporable_water_deficit = np.zeros(domain_size, dtype=np.float32)
            self.taw = np.zeros(domain_size, dtype=np.float32)  # Total available water
            self.raw = np.zeros(domain_size, dtype=np.float32)  # Readily available water
            self.bare_soil_evap = np.zeros(domain_size, dtype=np.float32)
            self.crop_etc = np.zeros(domain_size, dtype=np.float32)
            self.fraction_exposed_wetted = np.zeros(domain_size, dtype=np.float32)
            self.current_plant_height = np.zeros(domain_size, dtype=np.float32)
            self.kr = np.zeros(domain_size, dtype=np.float32)  # Evaporation reduction coefficient
            self.ke = np.zeros(domain_size, dtype=np.float32)  # Soil evaporation coefficient
            self.ks = np.zeros(domain_size, dtype=np.float32)  # Water stress coefficient

    def initialize(self, landuse_indices: NDArray[np.int32],
                   soil_indices: NDArray[np.int32],
                   elevation: NDArray[np.float32]) -> None:
        """Initialize calculator with spatial data"""
        # Validate array sizes
        if (len(landuse_indices) != self.domain_size or
                len(soil_indices) != self.domain_size or
                len(elevation) != self.domain_size):
            raise ValueError("Input array sizes must match domain size")

        self.landuse_indices = landuse_indices
        self.soil_indices = soil_indices
        self.elevation = elevation

        # Ensure parameters exist for all landuse types
        unique_landuse = np.unique(landuse_indices)
        for lu_code in unique_landuse:
            if lu_code not in self.params:
                self.params[lu_code] = self.default_params
                self.logger.warning(f"Using default parameters for landuse code {lu_code}")

        # Reset calculation arrays
        if self.method == "fao56_two_stage":
            self._reset_calculation_arrays()

    def _reset_calculation_arrays(self) -> None:
        """Reset all calculation arrays to initial values"""
        arrays_to_reset = [
            'evaporable_water_storage', 'evaporable_water_deficit',
            'taw', 'raw', 'bare_soil_evap', 'crop_etc',
            'fraction_exposed_wetted', 'current_plant_height',
            'kr', 'ke', 'ks'
        ]

        for array_name in arrays_to_reset:
            if hasattr(self, array_name):
                getattr(self, array_name).fill(0.0)

    def add_parameters(self, landuse_id: int, params: FAO56Parameters) -> None:
        """Add or update parameters for a landuse type"""
        self.params[landuse_id] = params

    def get_parameters(self, landuse_id: int) -> FAO56Parameters:
        """Get parameters for a landuse type, falling back to defaults if not found"""
        return self.params.get(landuse_id, self.default_params)

    def calculate_fao56(self, soil_storage: NDArray[np.float32],
                        soil_storage_max: NDArray[np.float32],
                        infiltration: NDArray[np.float32],
                        crop_etc: NDArray[np.float32]) -> None:
        """Calculate actual ET using FAO-56 method"""
        # Calculate total available water (TAW)
        taw = soil_storage_max

        # Calculate readily available water (RAW)
        raw = self.depletion_fraction * taw

        # Calculate current depletion
        depletion = soil_storage_max - soil_storage

        # Calculate stress coefficient Ks
        self.ks = np.ones_like(soil_storage)

        # Apply stress when depletion exceeds RAW
        stress_mask = depletion > raw
        self.ks[stress_mask] = (taw[stress_mask] - depletion[stress_mask]) / \
                               (taw[stress_mask] - raw[stress_mask])

        # Limit Ks to [0,1]
        self.ks = np.clip(self.ks, 0.0, 1.0)

        # Calculate actual ET
        self.actual_et = self.ks * crop_etc

        # Limit actual ET to available water
        self.actual_et = np.minimum(self.actual_et, soil_storage)

    def calculate_thornthwaite_mather(self, soil_storage: NDArray[np.float32],
                                      soil_storage_max: NDArray[np.float32],
                                      infiltration: NDArray[np.float32],
                                      crop_etc: NDArray[np.float32]) -> None:
        """Calculate actual ET using Thornthwaite-Mather method"""
        # Calculate soil moisture ratio
        moisture_ratio = np.clip(soil_storage / soil_storage_max, 0.0, 1.0)

        # Calculate actual ET as linear function of moisture
        self.actual_et = moisture_ratio * crop_etc

        # Limit actual ET to available water
        self.actual_et = np.minimum(self.actual_et, soil_storage)

    def calculate_fao56_two_stage(self,
                                  soil_storage: NDArray[np.float64],
                                  soil_storage_max: NDArray[np.float32],
                                  infiltration: NDArray[np.float32],
                                  reference_et0: NDArray[np.float64],
                                  kcb: NDArray[np.float32],
                                  landuse_indices: NDArray[np.int32],
                                  soil_group: NDArray[np.int32],
                                  awc: NDArray[np.float32],
                                  current_rooting_depth: NDArray[np.float32],
                                  is_growing_season: NDArray[np.bool_]) -> None:
        """Calculate actual ET using FAO-56 two-stage method"""
        # Update evaporable water storage with infiltration
        self.evaporable_water_storage += infiltration

        # Get TEW values for each cell using safe parameter lookup
        tew_values = np.array([self.get_parameters(i).tew for i in landuse_indices])

        # Update storage and calculate deficit
        for i in range(len(landuse_indices)):
            params = self.get_parameters(landuse_indices[i])
            self.evaporable_water_storage[i] = np.clip(
                self.evaporable_water_storage[i],
                0.0, params.tew
            )

        self.evaporable_water_deficit = np.maximum(
            0.0,
            tew_values - self.evaporable_water_storage
        )

        # Calculate evaporation reduction coefficient
        self.kr = np.ones_like(self.evaporable_water_deficit)
        for landuse_id, params in self.params.items():
            mask = landuse_indices == landuse_id
            if np.any(mask):
                deficit_mask = mask & (self.evaporable_water_deficit > params.rew)
                self.kr[deficit_mask] = np.clip(
                    (params.tew - self.evaporable_water_deficit[deficit_mask]) /
                    (params.tew - params.rew),
                    0.0, 1.0
                )
                self.kr[mask & (self.evaporable_water_deficit >= params.tew)] = 0.0

        # Calculate fraction of soil exposed
        self.fraction_exposed_wetted = np.ones_like(kcb)
        for landuse_id, params in self.params.items():
            mask = landuse_indices == landuse_id
            if np.any(mask):
                numerator = np.maximum(kcb[mask] - params.kcb_min, 0.0)
                denominator = params.kcb_mid - params.kcb_min

                if denominator > 0.0:
                    exponent = 1.0 + 0.5 * current_rooting_depth[mask]
                    fc = (numerator / denominator) ** exponent
                else:
                    fc = np.ones_like(numerator)

                fc = np.maximum(fc, params.min_fraction_covered_soil)
                self.fraction_exposed_wetted[mask] = np.clip(1.0 - fc, 0.05, 1.0)

        # Calculate soil evaporation coefficient
        kcb_max = self._calculate_max_kcb(kcb, current_rooting_depth)
        self.ke = np.minimum(
            self.kr * (kcb_max - kcb),
            self.fraction_exposed_wetted * kcb_max
        )

        # Calculate bare soil evaporation
        self.bare_soil_evap = reference_et0 * self.ke

        # Calculate water stress coefficient
        self.taw = current_rooting_depth * awc
        self.raw = self.taw * self.depletion_fraction

        soil_moisture_deficit = np.maximum(0.0, soil_storage_max - soil_storage)
        self.ks = np.ones_like(soil_moisture_deficit)

        stress_mask = soil_moisture_deficit >= self.raw
        raw_mask = stress_mask & (soil_moisture_deficit < self.taw)
        self.ks[raw_mask] = (
                (self.taw[raw_mask] - soil_moisture_deficit[raw_mask]) /
                (self.taw[raw_mask] - self.raw[raw_mask] + 1e-6)
        )
        self.ks[soil_moisture_deficit >= self.taw] = 0.0

        # Calculate crop ET
        self.crop_etc = np.minimum(
            reference_et0 * kcb * self.ks,
            soil_storage
        )

        # Calculate total actual ET
        self.actual_et = self.crop_etc + self.bare_soil_evap

    def _calculate_max_kcb(self, kcb: NDArray[np.float32],
                           plant_height: NDArray[np.float32],
                           wind_speed: float = 2.0,
                           relative_humidity: float = 55.0) -> NDArray[np.float32]:
        """Calculate maximum value of Kcb based on climate and height"""
        return np.where(
            plant_height > 0.0,
            kcb + 0.05 * np.maximum(0.0, (
                    0.04 * (wind_speed - 2.0) -
                    0.004 * (relative_humidity - 45.0) *
                    (plant_height / 3.0) ** 0.3
            )),
            kcb
        )


    def calculate(self,
                  soil_storage: NDArray[np.float64],
                  soil_storage_max: NDArray[np.float32],
                  infiltration: NDArray[np.float32],
                  crop_etc: NDArray[np.float32],
                  **kwargs) -> NDArray[np.float64]:
        """Calculate actual ET using selected method"""

        # Input validation
        if soil_storage is None or soil_storage_max is None:
            raise ValueError("soil_storage and soil_storage_max must not be None")

        if crop_etc is None:
            raise ValueError("crop_etc must not be None")

        # Ensure arrays are properly shaped
        if len(soil_storage) != self.domain_size or len(soil_storage_max) != self.domain_size:
            raise ValueError(f"Input arrays must match domain size {self.domain_size}")

        if self.method == "fao56":
            self.calculate_fao56(soil_storage, soil_storage_max, infiltration, crop_etc)
        elif self.method == "thornthwaite":
            self.calculate_thornthwaite_mather(soil_storage, soil_storage_max, infiltration, crop_etc)
        elif self.method == "fao56_two_stage":
            if not all(key in kwargs for key in ['reference_et0', 'kcb', 'landuse_indices',
                                                 'soil_group', 'awc', 'current_rooting_depth',
                                                 'is_growing_season']):
                raise ValueError("Missing required arguments for FAO56 two-stage method")

            self.calculate_fao56_two_stage(
                soil_storage, soil_storage_max, infiltration,
                kwargs['reference_et0'], kwargs['kcb'], kwargs['landuse_indices'],
                kwargs['soil_group'], kwargs['awc'], kwargs['current_rooting_depth'],
                kwargs['is_growing_season']
            )

        # Ensure we always return a valid array
        if self.actual_et is None:
            self.actual_et = np.zeros(self.domain_size, dtype=np.float32)

        return self.actual_et