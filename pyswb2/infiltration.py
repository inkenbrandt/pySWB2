from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

@dataclass
class InfiltrationParameters:
    """Parameters for infiltration calculations"""
    maximum_rate: float = 1.0  # Maximum infiltration rate in inches/day
    minimum_rate: float = 0.0  # Minimum infiltration rate in inches/day
    soil_storage_max: float = 8.0  # Maximum soil storage in inches
    cfgi_lower_limit: float = -40.0  # Lower limit for continuous frozen ground index
    cfgi_upper_limit: float = 83.0  # Upper limit for continuous frozen ground index
    cesspool_factor: float = 0.6  # Factor for cesspool infiltration
    disposal_well_factor: float = 0.7  # Factor for disposal well infiltration
    water_body_factor: float = 0.8  # Factor for water body recharge
    water_main_factor: float = 0.9  # Factor for water main leakage

class InfiltrationModule:
    """Module for handling infiltration calculations in SWB model"""
    
    def __init__(self, domain_size: int):
        """Initialize infiltration module
        
        Args:
            domain_size: Number of cells in model domain
        """
        # Initialize arrays for infiltration components
        self.domain_size = domain_size
        self.infiltration = np.zeros(domain_size, dtype=np.float32)
        self.net_infiltration = np.zeros(domain_size, dtype=np.float32)
        self.direct_infiltration = np.zeros(domain_size, dtype=np.float32)
        self.maximum_infiltration = np.zeros(domain_size, dtype=np.float32)
        
        # Initialize CFGI tracking
        self.cfgi = np.zeros(domain_size, dtype=np.float32)
        self.cfgi_lower_limit = np.full(domain_size, -40.0, dtype=np.float32)
        self.cfgi_upper_limit = np.full(domain_size, 83.0, dtype=np.float32)
        
        # Initialize additional infiltration sources
        self.cesspool_recharge = np.zeros(domain_size, dtype=np.float32)
        self.disposal_well_recharge = np.zeros(domain_size, dtype=np.float32)
        self.water_body_recharge = np.zeros(domain_size, dtype=np.float32)
        self.water_main_leakage = np.zeros(domain_size, dtype=np.float32)
        
        # Parameters for each soil type
        self.soil_params: Dict[int, InfiltrationParameters] = {}
        self.soil_indices: Optional[NDArray] = None
        
    def initialize(self, soil_indices: NDArray[np.int32],
                  soil_storage_max: NDArray[np.float32]) -> None:
        """Initialize module with soil data
        
        Args:
            soil_indices: Array mapping each cell to a soil type
            soil_storage_max: Maximum soil storage capacity for each cell
        """
        self.soil_indices = soil_indices
        self.soil_storage_max = soil_storage_max
        
    def add_soil_parameters(self, soil_id: int, params: InfiltrationParameters) -> None:
        """Add or update parameters for a soil type
        
        Args:
            soil_id: Unique identifier for the soil type
            params: InfiltrationParameters object containing soil-specific parameters
        """
        self.soil_params[soil_id] = params
        
        # Update CFGI limits for this soil type
        mask = (self.soil_indices == soil_id)
        self.cfgi_lower_limit[mask] = params.cfgi_lower_limit
        self.cfgi_upper_limit[mask] = params.cfgi_upper_limit
        
    def update_cfgi(self, cfgi_change: NDArray[np.float32]) -> None:
        """Update continuous frozen ground index
        
        Args:
            cfgi_change: Change in CFGI for each cell
        """
        self.cfgi += cfgi_change
        self.cfgi = np.clip(self.cfgi, self.cfgi_lower_limit, self.cfgi_upper_limit)
        
    def calculate_maximum_infiltration(self, precipitation: NDArray[np.float32],
                                    runoff: NDArray[np.float32],
                                    soil_moisture: NDArray[np.float32]) -> None:
        """Calculate maximum possible infiltration
        
        Args:
            precipitation: Precipitation depth for each cell
            runoff: Surface runoff for each cell
            soil_moisture: Current soil moisture for each cell
        """
        if self.soil_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        for soil_id, params in self.soil_params.items():
            mask = (self.soil_indices == soil_id)
            
            # Basic infiltration capacity
            self.maximum_infiltration[mask] = np.clip(
                precipitation[mask] - runoff[mask] + soil_moisture[mask],
                params.minimum_rate,
                params.maximum_rate
            )
            
            # Reduce infiltration for frozen ground
            frozen_factor = np.clip(1.0 - (self.cfgi[mask] / self.cfgi_upper_limit[mask]), 0.0, 1.0)
            self.maximum_infiltration[mask] *= frozen_factor
            
    def calculate_direct_infiltration(self) -> None:
        """Calculate direct infiltration from additional sources"""
        if self.soil_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        for soil_id, params in self.soil_params.items():
            mask = (self.soil_indices == soil_id)
            
            # Calculate contributions from each source
            self.direct_infiltration[mask] = (
                self.cesspool_recharge[mask] * params.cesspool_factor +
                self.disposal_well_recharge[mask] * params.disposal_well_factor +
                self.water_body_recharge[mask] * params.water_body_factor +
                self.water_main_leakage[mask] * params.water_main_factor
            )
            
    def calculate_net_infiltration(self, soil_storage: NDArray[np.float32],
                                 precipitation: NDArray[np.float32],
                                 runoff: NDArray[np.float32]) -> None:
        """Calculate final net infiltration
        
        Args:
            soil_storage: Current soil storage for each cell
            precipitation: Precipitation depth for each cell
            runoff: Surface runoff for each cell
        """
        if self.soil_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        # Calculate available storage capacity
        available_storage = self.soil_storage_max - soil_storage
        
        # Calculate potential infiltration
        self.calculate_maximum_infiltration(precipitation, runoff, soil_storage)
        self.calculate_direct_infiltration()
        
        # Total infiltration limited by available storage
        total_infiltration = self.maximum_infiltration + self.direct_infiltration
        self.net_infiltration = np.minimum(total_infiltration, available_storage)
        
        # Update infiltration tracking
        self.infiltration = self.net_infiltration - self.direct_infiltration

    def process_timestep(self, soil_storage: NDArray[np.float32],
                        precipitation: NDArray[np.float32],
                        runoff: NDArray[np.float32],
                        cfgi_change: NDArray[np.float32]) -> None:
        """Process all infiltration calculations for current timestep
        
        Args:
            soil_storage: Current soil storage for each cell
            precipitation: Precipitation depth for each cell
            runoff: Surface runoff for each cell
            cfgi_change: Change in continuous frozen ground index
        """
        # Update frozen ground conditions
        self.update_cfgi(cfgi_change)
        
        # Calculate infiltration components
        self.calculate_net_infiltration(soil_storage, precipitation, runoff)
