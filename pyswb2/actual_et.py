from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal
import numpy as np
from numpy.typing import NDArray


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
    
    def __init__(self, domain_size: int, method: Literal["fao56", "fao56_two_stage", "thornthwaite"] = "fao56"):
        """Initialize ET calculator
        
        Args:
            domain_size: Number of cells in model domain
            method: ET calculation method to use
        """
        if method not in ["fao56", "fao56_two_stage", "thornthwaite"]:
            raise ValueError("method must be one of: 'fao56', 'fao56_two_stage', 'thornthwaite'")
            
        self.method = method
        self.domain_size = domain_size
        
        # Initialize arrays
        self.actual_et = np.zeros(domain_size, dtype=np.float64)
        self.depletion_fraction = np.full(domain_size, 0.5, dtype=np.float32)
        
        # Additional arrays for FAO56 two-stage method
        if method == "fao56_two_stage":
            self.evaporable_water_storage = np.zeros(domain_size, dtype=np.float32)
            self.evaporable_water_deficit = np.zeros(domain_size, dtype=np.float32)
            self.taw = np.zeros(domain_size, dtype=np.float64)  # Total available water
            self.raw = np.zeros(domain_size, dtype=np.float64)  # Readily available water
            self.bare_soil_evap = np.zeros(domain_size, dtype=np.float32)
            self.crop_etc = np.zeros(domain_size, dtype=np.float32)
            self.fraction_exposed_wetted = np.zeros(domain_size, dtype=np.float32)
            self.current_plant_height = np.zeros(domain_size, dtype=np.float32)
            self.kr = np.zeros(domain_size, dtype=np.float64)  # Evaporation reduction coefficient
            self.ke = np.zeros(domain_size, dtype=np.float64)  # Soil evaporation coefficient
            self.ks = np.zeros(domain_size, dtype=np.float64)  # Water stress coefficient
            
        # Parameters for each landuse/soil type
        self.params: Dict[int, FAO56Parameters] = {}
        
    def _calculate_max_kcb(self,
                           kcb: NDArray[np.float32],
                           plant_height: NDArray[np.float32],
                           wind_speed: float = 2.0,
                           relative_humidity: float = 55.0,
                           ) -> NDArray[np.float32]:
        """Calculate maximum value of Kcb based on climate and height
        
        Args:
            wind_speed: Wind speed at 2m height (m/s)
            relative_humidity: Minimum relative humidity (%)
            kcb: Basal crop coefficient array
            plant_height: Plant height array (m)
            
        Returns:
            Array of maximum Kcb values
        """
        # Following FAO-56 equation for Kcb_max
        return np.where(
            plant_height > 0.0,
            kcb + 0.05 * np.maximum(0.0, (
                0.04 * (wind_speed - 2.0) - 
                0.004 * (relative_humidity - 45.0) * 
                (plant_height / 3.0) ** 0.3
            )),
            kcb
        )
        
    def _calculate_evaporation_reduction_coef(self, 
                                            landuse_indices: NDArray[np.int32],
                                            soil_group: NDArray[np.int32],
                                            evaporable_water_deficit: NDArray[np.float32]) -> NDArray[np.float64]:
        """Calculate evaporation reduction coefficient (Kr)"""
        kr = np.ones(self.domain_size, dtype=np.float64)
        
        for landuse_id, params in self.params.items():
            mask = landuse_indices == landuse_id
            if not np.any(mask):
                continue
                
            # Get REW and TEW for this landuse/soil combination
            rew = params.rew
            tew = params.tew
            
            # Calculate Kr based on FAO-56 equation
            deficit_mask = mask & (evaporable_water_deficit > rew)
            kr[deficit_mask] = np.clip(
                (tew - evaporable_water_deficit[deficit_mask]) / (tew - rew),
                0.0, 1.0
            )
            kr[mask & (evaporable_water_deficit >= tew)] = 0.0
            
        return kr
        
    def _calculate_exposed_soil_fraction(self,
                                       landuse_indices: NDArray[np.int32],
                                       kcb: NDArray[np.float32],
                                       current_plant_height: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate fraction of soil that is exposed and wetted"""
        few = np.ones(self.domain_size, dtype=np.float32)
        
        for landuse_id, params in self.params.items():
            mask = landuse_indices == landuse_id
            if not np.any(mask):
                continue
                
            # Calculate vegetation cover fraction (fc)
            numerator = np.maximum(kcb[mask] - params.kcb_min, 0.0)
            denominator = params.kcb_mid - params.kcb_min
            
            if denominator > 0.0:
                exponent = 1.0 + 0.5 * current_plant_height[mask]
                fc = (numerator / denominator) ** exponent
            else:
                fc = np.ones_like(numerator)
                
            fc = np.maximum(fc, params.min_fraction_covered_soil)
            few[mask] = np.clip(1.0 - fc, 0.05, 1.0)
            
        return few
        
    def _update_plant_height(self,
                            landuse_indices: NDArray[np.int32],
                            is_growing_season: NDArray[np.bool_],
                            kcb: NDArray[np.float32]) -> NDArray[np.float32]:
        """Update plant height based on growing season and Kcb"""
        plant_height = np.full(self.domain_size, 0.1, dtype=np.float32)  # Minimum height
        
        for landuse_id, params in self.params.items():
            mask = landuse_indices == landuse_id
            if not np.any(mask):
                continue
                
            growing_mask = mask & is_growing_season
            if not np.any(growing_mask):
                continue
                
            numerator = kcb[growing_mask] - params.kcb_min
            denominator = params.kcb_mid - params.kcb_min
            
            height = numerator / denominator * params.mean_plant_height
            plant_height[growing_mask] = np.clip(height, 0.1, 10.0)
            
        return plant_height

    def calculate_fao56(self, soil_storage: NDArray[np.float32],
                         soil_storage_max: NDArray[np.float32],
                         infiltration: NDArray[np.float32],
                         crop_etc: NDArray[np.float32]) -> None:
        """Calculate actual ET using FAO-56 method

        This implements the soil water stress coefficient (Ks) approach from
        FAO-56 Chapter 8. When soil moisture drops below a threshold, actual ET
        is reduced proportionally.
        """
        # Calculate total available water (TAW)
        taw = soil_storage_max

        # Calculate readily available water (RAW)
        raw = self.depletion_fraction * taw

        # Calculate current depletion
        depletion = soil_storage_max - soil_storage

        # Calculate stress coefficient Ks
        ks = np.ones_like(soil_storage)

        # Apply stress when depletion exceeds RAW
        stress_mask = depletion > raw
        ks[stress_mask] = (taw[stress_mask] - depletion[stress_mask]) / \
                          (taw[stress_mask] - raw[stress_mask])

        # Limit Ks to [0,1]
        ks = np.clip(ks, 0.0, 1.0)

        # Calculate actual ET
        self.actual_et = ks * crop_etc

        # Limit actual ET to available water
        self.actual_et = np.minimum(self.actual_et, soil_storage)

    def calculate_thornthwaite_mather(self, soil_storage: NDArray[np.float32],
                                       soil_storage_max: NDArray[np.float32],
                                       infiltration: NDArray[np.float32],
                                       crop_etc: NDArray[np.float32]) -> None:
        """Calculate actual ET using Thornthwaite-Mather method

        This implements the original Thornthwaite-Mather approach where
        actual ET declines linearly with soil moisture.
        """
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
        """Calculate actual ET using FAO-56 two-stage method
        
        Args:
            soil_storage: Current soil moisture storage
            soil_storage_max: Maximum soil moisture storage
            infiltration: Infiltration amount
            reference_et0: Reference ET
            kcb: Basal crop coefficient
            landuse_indices: Array mapping cells to landuse types
            soil_group: Array mapping cells to soil groups
            awc: Available water capacity
            current_rooting_depth: Current root depth
            is_growing_season: Boolean array indicating growing season
        """
        # Update plant heights
        self.current_plant_height = self._update_plant_height(
            landuse_indices, is_growing_season, kcb
        )
        
        # Calculate maximum Kcb
        kcb_max = self._calculate_max_kcb(
            kcb=kcb,
            plant_height=self.current_plant_height
        )
        
        # Update evaporable water storage and deficit
        self.evaporable_water_storage += infiltration
        for landuse_id, params in self.params.items():
            mask = landuse_indices == landuse_id
            if np.any(mask):
                self.evaporable_water_storage[mask] = np.clip(
                    self.evaporable_water_storage[mask],
                    0.0, params.tew
                )
        self.evaporable_water_deficit = np.maximum(
            0.0,
            np.array([self.params[i].tew for i in landuse_indices]) - 
            self.evaporable_water_storage
        )
        
        # Calculate evaporation reduction coefficient
        self.kr = self._calculate_evaporation_reduction_coef(
            landuse_indices, soil_group, self.evaporable_water_deficit
        )
        
        # Calculate exposed soil fraction
        self.fraction_exposed_wetted = self._calculate_exposed_soil_fraction(
            landuse_indices, kcb, self.current_plant_height
        )
        
        # Calculate soil evaporation coefficient
        self.ke = np.minimum(
            self.kr * (kcb_max - kcb),
            self.fraction_exposed_wetted * kcb_max
        )
        
        # Calculate bare soil evaporation
        self.bare_soil_evap = reference_et0 * self.ke
        
        # Update soil moisture and calculate water stress coefficient
        interim_storage = np.clip(
            soil_storage + infiltration - self.bare_soil_evap,
            0.0, soil_storage_max
        )
        
        # Adjust bare soil evaporation based on available water
        self.bare_soil_evap = np.clip(
            soil_storage + infiltration - interim_storage,
            0.0, soil_storage_max
        )
        
        # Calculate total and readily available water
        self.taw = current_rooting_depth * awc
        
        # Update depletion fractions and RAW
        for i in range(self.domain_size):
            self.depletion_fraction[i] = self._adjust_depletion_fraction(
                self.depletion_fraction[i],
                reference_et0[i]
            )
        self.raw = self.taw * self.depletion_fraction
        
        # Calculate soil moisture deficit and stress coefficient
        soil_moisture_deficit = np.maximum(0.0, soil_storage_max - interim_storage)
        
        # Calculate water stress coefficient
        self.ks = np.ones_like(soil_moisture_deficit, dtype=np.float64)
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
            interim_storage
        )
        
        # Calculate total actual ET
        self.actual_et = self.crop_etc + self.bare_soil_evap
        
    def calculate(self,
                 soil_storage: NDArray[np.float64],
                 soil_storage_max: NDArray[np.float32],
                 infiltration: NDArray[np.float32],
                 crop_etc: NDArray[np.float32],
                 **kwargs) -> NDArray[np.float64]:
        """Calculate actual ET using selected method"""
        if self.method == "fao56":
            self.calculate_fao56(soil_storage, soil_storage_max, infiltration, crop_etc)
        elif self.method == "thornthwaite":
            self.calculate_thornthwaite_mather(soil_storage, soil_storage_max, infiltration, crop_etc)
        elif self.method == "fao56_two_stage":
            required_args = ['reference_et0', 'kcb', 'landuse_indices', 'soil_group',
                           'awc', 'current_rooting_depth', 'is_growing_season']
            missing_args = [arg for arg in required_args if arg not in kwargs]
            if missing_args:
                raise ValueError(f"Missing required arguments for FAO56 two-stage method: {missing_args}")
                
            self.calculate_fao56_two_stage(
                soil_storage, soil_storage_max, infiltration,
                kwargs['reference_et0'], kwargs['kcb'], kwargs['landuse_indices'],
                kwargs['soil_group'], kwargs['awc'], kwargs['current_rooting_depth'],
                kwargs['is_growing_season']
            )
            
        return self.actual_et




