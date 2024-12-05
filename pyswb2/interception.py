from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

@dataclass
class GashParameters:
    """Parameters for Gash interception model"""
    canopy_storage_capacity: float = 0.0  # Canopy storage capacity in inches
    trunk_storage_capacity: float = 0.0   # Trunk storage capacity in inches
    stemflow_fraction: float = 0.0        # Fraction of precipitation that becomes stemflow
    interception_storage_max_growing: float = 0.1    # Maximum interception storage during growing season
    interception_storage_max_nongrowing: float = 0.1 # Maximum interception storage during non-growing season

@dataclass
class BucketParameters:
    """Parameters for Bucket interception model"""
    growing_season_a: float = 0.0     # 'a' coefficient for growing season
    growing_season_b: float = 0.0     # 'b' coefficient for growing season
    growing_season_n: float = 1.0     # 'n' exponent for growing season
    nongrowing_season_a: float = 0.0  # 'a' coefficient for non-growing season
    nongrowing_season_b: float = 0.0  # 'b' coefficient for non-growing season
    nongrowing_season_n: float = 1.0  # 'n' exponent for non-growing season
    interception_storage_max_growing: float = 0.1    # Maximum interception storage during growing season
    interception_storage_max_nongrowing: float = 0.1 # Maximum interception storage during non-growing season

class InterceptionModule:
    """Module for calculating canopy interception using either Gash or Bucket method"""
    
    def __init__(self, domain_size: int, method: str = "gash"):
        """Initialize interception module
        
        Args:
            domain_size: Number of cells in model domain
            method: Interception calculation method ('gash' or 'bucket')
        """
        if method not in ["gash", "bucket"]:
            raise ValueError("method must be either 'gash' or 'bucket'")
            
        self.method = method
        self.domain_size = domain_size
        
        # Initialize arrays
        self.interception = np.zeros(domain_size, dtype=np.float32)
        self.interception_storage = np.zeros(domain_size, dtype=np.float32)
        self.interception_storage_max = np.zeros(domain_size, dtype=np.float32)
        self.evap_to_rain_ratio = np.ones(domain_size, dtype=np.float32)
        self.canopy_cover_fraction = np.zeros(domain_size, dtype=np.float32)
        
        # Parameters for each landuse type
        self.gash_params: Dict[int, GashParameters] = {}
        self.bucket_params: Dict[int, BucketParameters] = {}
        self.landuse_indices: Optional[NDArray] = None
        
    def initialize(self, landuse_indices: NDArray[np.int32], 
                  canopy_cover: NDArray[np.float32],
                  evap_to_rain_ratio: Optional[NDArray[np.float32]] = None) -> None:
        """Initialize module with landuse and canopy data
        
        Args:
            landuse_indices: Array mapping each cell to a landuse type
            canopy_cover: Canopy cover fraction for each cell
            evap_to_rain_ratio: Optional ratio of evaporation to rainfall rate
        """
        self.landuse_indices = landuse_indices
        self.canopy_cover_fraction = canopy_cover
        
        if evap_to_rain_ratio is not None:
            self.evap_to_rain_ratio = evap_to_rain_ratio
            
        # Set initial storage max based on season
        self._update_storage_max_by_season(np.zeros_like(landuse_indices, dtype=bool))
        
    def add_gash_parameters(self, landuse_id: int, params: GashParameters) -> None:
        """Add or update Gash parameters for a landuse type
        
        Args:
            landuse_id: Unique identifier for the landuse type
            params: GashParameters object containing landuse-specific parameters
        """
        self.gash_params[landuse_id] = params
        
    def add_bucket_parameters(self, landuse_id: int, params: BucketParameters) -> None:
        """Add or update Bucket parameters for a landuse type
        
        Args:
            landuse_id: Unique identifier for the landuse type
            params: BucketParameters object containing landuse-specific parameters
        """
        self.bucket_params[landuse_id] = params
        
    def calculate_interception(self, rainfall: NDArray[np.float32], 
                             fog: NDArray[np.float32],
                             is_growing_season: NDArray[np.bool_]) -> None:
        """Calculate interception for current timestep
        
        Args:
            rainfall: Rainfall depth for each cell
            fog: Fog deposition for each cell
            is_growing_season: Boolean array indicating growing season status
        """
        if self.landuse_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        # Update storage max based on season
        self._update_storage_max_by_season(is_growing_season)
        
        # Calculate potential interception
        if self.method == "gash":
            self._calculate_gash_interception(rainfall, fog, is_growing_season)
        else:
            self._calculate_bucket_interception(rainfall, fog, is_growing_season)
            
        # Update storage tracking
        self._update_storage(rainfall + fog)

    def _calculate_gash_interception(self, rainfall: NDArray[np.float32],
                                     fog: NDArray[np.float32],
                                     is_growing_season: NDArray[np.bool_]) -> None:
        """Calculate interception using Gash method"""
        total_input = rainfall + fog

        for landuse_id, params in self.gash_params.items():
            mask = (self.landuse_indices == landuse_id)
            if not np.any(mask):
                continue

            # Calculate saturation precipitation
            p_sat = self._calc_precipitation_at_saturation(
                self.evap_to_rain_ratio[mask],
                params.canopy_storage_capacity,
                self.canopy_cover_fraction[mask]
            )

            # Get masked inputs
            cell_input = total_input[mask]

            # Create masks for different conditions within this landuse type
            below_sat = cell_input < p_sat
            above_trunk = cell_input > (params.trunk_storage_capacity /
                                        (params.stemflow_fraction + 1e-6))
            between = ~(below_sat | above_trunk)

            # Calculate interception for cells below saturation
            masked_indices = mask & below_sat
            self.interception[masked_indices] = (
                    self.canopy_cover_fraction[masked_indices] * total_input[masked_indices]
            )

            # Calculate interception for cells above trunk storage capacity
            masked_indices = mask & above_trunk
            self.interception[masked_indices] = (
                    self.canopy_cover_fraction[masked_indices] * p_sat[above_trunk] +
                    self.canopy_cover_fraction[masked_indices] *
                    self.evap_to_rain_ratio[masked_indices] *
                    (total_input[masked_indices] - p_sat[above_trunk]) +
                    params.trunk_storage_capacity
            )

            # Calculate interception for cells between saturation and trunk storage
            masked_indices = mask & between
            self.interception[masked_indices] = (
                    self.canopy_cover_fraction[masked_indices] * p_sat[between] +
                    self.canopy_cover_fraction[masked_indices] *
                    self.evap_to_rain_ratio[masked_indices] *
                    (total_input[masked_indices] - p_sat[between]) +
                    params.stemflow_fraction * total_input[masked_indices]
            )
            
    def _calculate_bucket_interception(self, rainfall: NDArray[np.float32],
                                     fog: NDArray[np.float32],
                                     is_growing_season: NDArray[np.bool_]) -> None:
        """Calculate interception using Bucket method"""
        total_input = rainfall + fog
        
        for landuse_id, params in self.bucket_params.items():
            mask = (self.landuse_indices == landuse_id)
            growing_mask = mask & is_growing_season
            nongrowing_mask = mask & ~is_growing_season
            
            # Calculate potential interception for growing season
            if np.any(growing_mask):
                self.interception[growing_mask] = (
                    params.growing_season_a +
                    params.growing_season_b * total_input[growing_mask] **
                    params.growing_season_n
                ) * self.canopy_cover_fraction[growing_mask]
                
            # Calculate potential interception for non-growing season
            if np.any(nongrowing_mask):
                self.interception[nongrowing_mask] = (
                    params.nongrowing_season_a +
                    params.nongrowing_season_b * total_input[nongrowing_mask] **
                    params.nongrowing_season_n
                ) * self.canopy_cover_fraction[nongrowing_mask]
                
    def _calc_precipitation_at_saturation(self, e_div_p: NDArray[np.float32],
                                        canopy_storage: float,
                                        canopy_fraction: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate precipitation needed to saturate canopy"""
        mask = (canopy_fraction > 0.0) & (canopy_storage > 0.0)
        p_sat = np.zeros_like(e_div_p)
        
        p_sat[mask] = -(canopy_storage / (canopy_fraction[mask] * e_div_p[mask])) * \
                      np.log(1.0 - e_div_p[mask])
        
        return p_sat
        
    def _update_storage_max_by_season(self, is_growing_season: NDArray[np.bool_]) -> None:
        """Update maximum storage capacity based on growing season"""
        if self.method == "gash":
            params_dict = self.gash_params
        else:
            params_dict = self.bucket_params
            
        for landuse_id, params in params_dict.items():
            mask = (self.landuse_indices == landuse_id)
            self.interception_storage_max[mask & is_growing_season] = (
                params.interception_storage_max_growing
            )
            self.interception_storage_max[mask & ~is_growing_season] = (
                params.interception_storage_max_nongrowing
            )
            
    def _update_storage(self, total_input: NDArray[np.float32]) -> None:
        """Update interception storage tracking"""
        # Add new interception to storage
        self.interception_storage += self.interception
        
        # Limit to maximum storage
        excess = np.maximum(0.0, self.interception_storage - self.interception_storage_max)
        self.interception_storage = np.minimum(
            self.interception_storage, 
            self.interception_storage_max
        )
        
        # Adjust interception amount for excess
        self.interception = np.minimum(self.interception, total_input)
        self.interception -= excess
