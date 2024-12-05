from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

@dataclass
class SoilParameters:
    """Parameters for soil calculations"""
    awc: float = 0.2  # Available water capacity (inches/foot)
    field_capacity: float = 0.3  # Field capacity (fraction)
    wilting_point: float = 0.1  # Wilting point (fraction)
    hydraulic_conductivity: float = 1.0  # Saturated hydraulic conductivity (inches/day)
    suction_head: float = 8.0  # Suction head at wetting front (inches)
    bulk_density: float = 1.5  # Soil bulk density (g/cm³)
    organic_matter: float = 0.02  # Organic matter content (fraction)

class SoilModule:
    """Module for handling soil water calculations in SWB model"""
    
    def __init__(self, domain_size: int, method: str = "integrated"):
        """Initialize soil module
        
        Args:
            domain_size: Number of cells in model domain
            method: AWC calculation method ('integrated' or 'gridded')
        """
        if method not in ["integrated", "gridded"]:
            raise ValueError("method must be either 'integrated' or 'gridded'")
            
        self.method = method
        self.domain_size = domain_size
        
        # Initialize main arrays
        self.soil_storage = np.zeros(domain_size, dtype=np.float32)
        self.soil_storage_max = np.zeros(domain_size, dtype=np.float32)
        self.delta_soil_storage = np.zeros(domain_size, dtype=np.float32)
        self.available_water_content = np.zeros(domain_size, dtype=np.float32)
        
        # Mass balance components
        self.infiltration = np.zeros(domain_size, dtype=np.float32)
        self.net_infiltration = np.zeros(domain_size, dtype=np.float32)
        self.direct_soil_moisture = np.zeros(domain_size, dtype=np.float32)
        self.actual_et_soil = np.zeros(domain_size, dtype=np.float32)
        
        # Soil properties by type
        self.soil_params: Dict[int, SoilParameters] = {}
        self.soil_indices: Optional[NDArray] = None
        
    def initialize(self, soil_indices: NDArray[np.int32],
                  rooting_depth: NDArray[np.float32],
                  awc_grid: Optional[NDArray[np.float32]] = None) -> None:
        """Initialize module with soil data
        
        Args:
            soil_indices: Array mapping each cell to a soil type
            rooting_depth: Rooting depth for each cell (feet)
            awc_grid: Optional gridded AWC data (if using gridded method)
        """
        self.soil_indices = soil_indices
        
        if self.method == "gridded" and awc_grid is not None:
            self.available_water_content = awc_grid
        else:
            self._calculate_integrated_awc(rooting_depth)
            
        # Calculate maximum soil storage based on AWC and rooting depth
        self.soil_storage_max = self.available_water_content * rooting_depth
        
    def _calculate_integrated_awc(self, rooting_depth: NDArray[np.float32]) -> None:
        """Calculate depth-integrated available water content
        
        Args:
            rooting_depth: Rooting depth for each cell (feet)
        """
        for soil_id, params in self.soil_params.items():
            mask = (self.soil_indices == soil_id)
            if not np.any(mask):
                continue
                
            # Basic AWC calculation - could be enhanced with horizon data
            self.available_water_content[mask] = (
                params.field_capacity - params.wilting_point
            ) * 12.0  # Convert to inches/foot
            
    def add_soil_parameters(self, soil_id: int, params: SoilParameters) -> None:
        """Add or update parameters for a soil type
        
        Args:
            soil_id: Unique identifier for the soil type
            params: SoilParameters object containing soil-specific parameters
        """
        self.soil_params[soil_id] = params
        
    def calculate_soil_moisture(self, precipitation: NDArray[np.float32],
                              actual_et: NDArray[np.float32],
                              reference_et0: NDArray[np.float32]) -> None:
        """Calculate soil moisture balance
        
        Args:
            precipitation: Net precipitation after interception
            actual_et: Actual evapotranspiration
            reference_et0: Reference ET
        """
        if self.soil_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        # Calculate infiltration for each soil type
        self._calculate_infiltration(precipitation)
        
        # Add direct soil moisture inputs (e.g., septic discharge)
        self._calculate_direct_soil_moisture()
        
        # Calculate soil mass balance
        self._calculate_soil_mass_balance(actual_et, reference_et0)
        
    def _calculate_infiltration(self, precipitation: NDArray[np.float32]) -> None:
        """Calculate infiltration based on soil properties
        
        Args:
            precipitation: Net precipitation available for infiltration
        """
        for soil_id, params in self.soil_params.items():
            mask = (self.soil_indices == soil_id)
            if not np.any(mask):
                continue
                
            # Simple infiltration based on hydraulic conductivity
            self.infiltration[mask] = np.minimum(
                precipitation[mask],
                params.hydraulic_conductivity
            )
            
            # Calculate net infiltration considering soil storage capacity
            available_storage = self.soil_storage_max[mask] - self.soil_storage[mask]
            self.net_infiltration[mask] = np.minimum(
                self.infiltration[mask],
                available_storage
            )
            
    def _calculate_direct_soil_moisture(self) -> None:
        """Calculate direct soil moisture additions"""
        # Implementation would depend on specific sources like septic systems
        # For now, assuming direct_soil_moisture is updated externally
        pass
        
    def _calculate_soil_mass_balance(self, actual_et: NDArray[np.float32],
                                   reference_et0: NDArray[np.float32]) -> None:
        """Calculate soil moisture mass balance
        
        Args:
            actual_et: Actual evapotranspiration
            reference_et0: Reference ET
        """
        # Store previous storage for delta calculation
        previous_storage = self.soil_storage.copy()
        
        # Add infiltration and direct moisture
        self.soil_storage += (self.net_infiltration + self.direct_soil_moisture)
        
        # Calculate actual ET from soil
        self.actual_et_soil = np.minimum(
            actual_et,
            self.soil_storage
        )
        self.soil_storage -= self.actual_et_soil
        
        # Ensure storage doesn't exceed maximum
        excess = np.maximum(0.0, self.soil_storage - self.soil_storage_max)
        self.soil_storage = np.minimum(self.soil_storage, self.soil_storage_max)
        
        # Calculate change in storage
        self.delta_soil_storage = self.soil_storage - previous_storage
        
    def get_soil_moisture_deficit(self) -> NDArray[np.float32]:
        """Calculate soil moisture deficit
        
        Returns:
            Array of soil moisture deficits
        """
        return np.maximum(0.0, self.soil_storage_max - self.soil_storage)
        
    def get_relative_soil_moisture(self) -> NDArray[np.float32]:
        """Calculate relative soil moisture content
        
        Returns:
            Array of relative soil moisture (0-1)
        """
        return np.clip(self.soil_storage / self.soil_storage_max, 0.0, 1.0)

    def process_timestep(self, date: datetime,
                        precipitation: NDArray[np.float32],
                        actual_et: NDArray[np.float32],
                        reference_et0: NDArray[np.float32]) -> None:
        """Process all soil calculations for current timestep
        
        Args:
            date: Current simulation date
            precipitation: Net precipitation after interception
            actual_et: Actual evapotranspiration
            reference_et0: Reference ET
        """
        # Calculate soil moisture components
        self.calculate_soil_moisture(precipitation, actual_et, reference_et0)
