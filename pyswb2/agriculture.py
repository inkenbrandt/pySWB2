from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

@dataclass
class CropParameters:
    """Crop-specific parameters for agricultural calculations"""
    gdd_base: float = 50.0  # Base temperature for GDD calculation
    gdd_max: float = 86.0   # Maximum temperature for GDD calculation
    gdd_reset_date: int = 365  # Day of year to reset GDD
    growing_season_start: Optional[int] = None  # Starting day of year
    growing_season_end: Optional[int] = None    # Ending day of year
    gdd_growing_season_start: Optional[float] = None  # GDD threshold to start growing season
    killing_frost_temp: Optional[float] = None  # Temperature threshold to end growing season
    initial_root_depth: float = 50.0  # Initial rooting depth in mm
    max_root_depth: float = 1500.0    # Maximum rooting depth in mm
    root_growth_rate: float = 2.5     # Root growth rate in mm/day
    crop_coefficient: float = 1.0     # Base crop coefficient

class AgricultureModule:
    """Module for handling agricultural calculations in SWB model"""

    def __init__(self, domain_size: int):
        """Initialize agriculture module

        Args:
            domain_size: Number of cells in model domain
        """
        # Initialize arrays for each cell
        self.domain_size = domain_size
        self.gdd = np.zeros(domain_size, dtype=np.float32)
        self.is_growing_season = np.zeros(domain_size, dtype=bool)
        self.root_depth = np.zeros(domain_size, dtype=np.float32)
        self.crop_coefficient = np.ones(domain_size, dtype=np.float32)

        # Store parameters for each crop type
        self.crop_params: Dict[int, CropParameters] = {}

        # Track current values
        self.current_date: Optional[datetime] = None
        self.landuse_indices: Optional[NDArray] = None

    
    def add_crop_parameters(self, crop_id: int, params: CropParameters) -> None:
        """Add or update parameters for a crop type
        
        Args:
            crop_id: Unique identifier for the crop type
            params: CropParameters object containing crop-specific parameters
        """
        self.crop_params[crop_id] = params

    def initialize(self, landuse_indices: NDArray[np.int32]) -> None:
        """Initialize module with landuse data

        Args:
            landuse_indices: Array mapping each cell to a crop/landuse type
        """
        self.landuse_indices = landuse_indices

        # Initialize root depths
        for crop_id, params in self.crop_params.items():
            mask = (landuse_indices == crop_id)
            self.root_depth[mask] = params.initial_root_depth
            self.crop_coefficient[mask] = params.crop_coefficient

    def calculate_growing_degree_days(self, tmean: NDArray[np.float32], 
                                    tmin: NDArray[np.float32],
                                    tmax: NDArray[np.float32]) -> None:
        """Calculate growing degree days using modified GDD method
        
        Args:
            tmean: Mean daily temperature array
            tmin: Minimum daily temperature array
            tmax: Maximum daily temperature array
        """
        if self.landuse_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        for crop_id, params in self.crop_params.items():
            mask = (self.landuse_indices == crop_id)
            
            # Reset GDD if needed
            if self.current_date and self.current_date.timetuple().tm_yday == params.gdd_reset_date:
                self.gdd[mask] = 0.0
            
            # Calculate GDD using modified method
            tmax_adj = np.minimum(tmax[mask], params.gdd_max)
            tmin_adj = np.maximum(tmin[mask], params.gdd_base)
            tmean_adj = (tmax_adj + tmin_adj) / 2.0
            
            self.gdd[mask] += np.maximum(
                tmean_adj - params.gdd_base,
                0.0
            )

    def update_root_depth(self) -> None:
        """Update root depths based on growing season status"""
        if self.landuse_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        for crop_id, params in self.crop_params.items():
            mask = (self.landuse_indices == crop_id) & self.is_growing_season
            
            # Increase root depth during growing season
            self.root_depth[mask] = np.minimum(
                self.root_depth[mask] + params.root_growth_rate,
                params.max_root_depth
            )
    
    def update_crop_coefficients(self) -> None:
        """Update crop coefficients based on growing season status"""
        if self.landuse_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        for crop_id, params in self.crop_params.items():
            mask = (self.landuse_indices == crop_id)
            
            # Set coefficient based on growing season
            self.crop_coefficient[mask] = np.where(
                self.is_growing_season[mask],
                params.crop_coefficient,
                0.1  # Minimal coefficient outside growing season
            )
    
    def process_daily(self, date: datetime, tmean: NDArray[np.float32],
                     tmin: NDArray[np.float32], tmax: NDArray[np.float32]) -> None:
        """Process all daily agricultural calculations
        
        Args:
            date: Current simulation date
            tmean: Mean daily temperature array
            tmin: Minimum daily temperature array
            tmax: Maximum daily temperature array
        """
        self.current_date = date
        
        # Update all components
        self.calculate_growing_degree_days(tmean, tmin, tmax)
        self.update_growing_season(tmean)
        self.update_root_depth()
        self.update_crop_coefficients()

    def update_growing_season(self, tmean: NDArray[np.float32]) -> None:
        """Update growing season status for each cell

        Args:
            tmean: Mean daily temperature array
        """
        if self.landuse_indices is None:
            raise RuntimeError("Module must be initialized before calculations")

        if self.current_date is None:
            raise RuntimeError("Current date must be set before updating growing season")

        current_doy = self.current_date.timetuple().tm_yday

        for crop_id, params in self.crop_params.items():
            mask = (self.landuse_indices == crop_id)

            # Check GDD-based start
            if params.gdd_growing_season_start is not None:
                start_mask = mask & ~self.is_growing_season
                self.is_growing_season[start_mask] = (
                        self.gdd[start_mask] >= params.gdd_growing_season_start
                )

            # Check DOY-based start
            elif params.growing_season_start is not None:
                start_mask = mask & ~self.is_growing_season
                self.is_growing_season[start_mask] = (
                        current_doy >= params.growing_season_start
                )

            # Check temperature-based end
            if params.killing_frost_temp is not None:
                end_mask = mask & self.is_growing_season
                self.is_growing_season[end_mask] &= (
                        tmean[end_mask] > params.killing_frost_temp
                )

            # Check DOY-based end
            elif params.growing_season_end is not None:
                end_mask = mask & self.is_growing_season
                self.is_growing_season[end_mask] &= (
                        current_doy <= params.growing_season_end
                )