from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
import numpy as np
from numpy.typing import NDArray

@dataclass
class RunoffParameters:
    """Parameters for runoff calculations"""
    curve_number: float = 70.0  # SCS curve number
    initial_abstraction_ratio: float = 0.2  # Initial abstraction ratio
    depression_storage: float = 0.1  # Depression storage capacity (inches)
    impervious_fraction: float = 0.0  # Fraction of impervious surface
    storm_drain_capture_fraction: float = 0.0  # Fraction captured by storm drains
    minimum_runoff: float = 0.01  # Minimum runoff threshold (inches)

class RunoffModule:
    """Module for handling runoff calculations in SWB model"""
    
    def __init__(self, domain_size: int, method: Literal["curve_number", "gridded"] = "curve_number"):
        """Initialize runoff module
        
        Args:
            domain_size: Number of cells in model domain
            method: Runoff calculation method
        """
        if method not in ["curve_number", "gridded"]:
            raise ValueError("method must be either 'curve_number' or 'gridded'")
            
        self.method = method
        self.domain_size = domain_size
        
        # Initialize runoff arrays
        self.runoff = np.zeros(domain_size, dtype=np.float32)
        self.runoff_outside = np.zeros(domain_size, dtype=np.float32)
        self.storm_drain_capture = np.zeros(domain_size, dtype=np.float32)
        self.depression_storage = np.zeros(domain_size, dtype=np.float32)
        
        # Initialize curve number arrays
        self.cn_current = np.zeros(domain_size, dtype=np.float32)
        self.cn_dry = np.zeros(domain_size, dtype=np.float32)  # ARC I
        self.cn_normal = np.zeros(domain_size, dtype=np.float32)  # ARC II
        self.cn_wet = np.zeros(domain_size, dtype=np.float32)  # ARC III
        
        # Track antecedent conditions
        self.prev_5day_precip = np.zeros(domain_size, dtype=np.float32)
        
        # Parameters for each landuse type
        self.params: Dict[int, RunoffParameters] = {}
        self.landuse_indices: Optional[NDArray] = None
        
        # Gridded runoff parameters if using gridded method
        self.runoff_zones: Optional[NDArray] = None
        self.runoff_ratios: Optional[NDArray] = None
        self.monthly_ratios: Optional[Dict[int, List[float]]] = None
        
    def initialize(self, landuse_indices: NDArray[np.int32],
                  runoff_zones: Optional[NDArray[np.int32]] = None,
                  monthly_ratios: Optional[Dict[int, List[float]]] = None) -> None:
        """Initialize module with landuse data
        
        Args:
            landuse_indices: Array mapping each cell to a landuse type
            runoff_zones: Optional array mapping cells to runoff zones
            monthly_ratios: Optional dictionary of monthly adjustment ratios by zone
        """
        self.landuse_indices = landuse_indices
        
        # Initialize curve numbers for each landuse type
        for landuse_id, params in self.params.items():
            mask = (landuse_indices == landuse_id)
            if not np.any(mask):
                continue
                
            # Set initial curve numbers
            self.cn_normal[mask] = params.curve_number
            self.cn_dry[mask] = self._adjust_cn_for_condition(
                params.curve_number, "dry"
            )
            self.cn_wet[mask] = self._adjust_cn_for_condition(
                params.curve_number, "wet"
            )
            
            # Set initial condition to normal
            self.cn_current[mask] = self.cn_normal[mask]
            
            # Set depression storage
            self.depression_storage[mask] = params.depression_storage
            
        # Initialize gridded parameters if using gridded method
        if self.method == "gridded":
            if runoff_zones is None or monthly_ratios is None:
                raise ValueError("runoff_zones and monthly_ratios required for gridded method")
                
            self.runoff_zones = runoff_zones
            self.monthly_ratios = monthly_ratios
            self.runoff_ratios = np.ones(self.domain_size, dtype=np.float32)
            
    def add_parameters(self, landuse_id: int, params: RunoffParameters) -> None:
        """Add or update parameters for a landuse type
        
        Args:
            landuse_id: Unique identifier for the landuse type
            params: RunoffParameters object containing landuse-specific parameters
        """
        self.params[landuse_id] = params
        
    def _adjust_cn_for_condition(self, cn: float, condition: Literal["dry", "wet"]) -> float:
        """Adjust curve number for antecedent runoff condition
        
        Args:
            cn: Base curve number (ARC II)
            condition: Target condition ("dry" for ARC I or "wet" for ARC III)
            
        Returns:
            Adjusted curve number
        """
        if condition == "dry":
            return 4.2 * cn / (10 - 0.058 * cn)
        else:  # wet
            return 23 * cn / (10 + 0.13 * cn)
            
    def update_antecedent_conditions(self, precipitation: NDArray[np.float32],
                                   growing_season: NDArray[np.bool_]) -> None:
        """Update antecedent conditions based on recent precipitation
        
        Args:
            precipitation: Current precipitation array
            growing_season: Boolean array indicating growing season status
        """
        # Update 5-day precipitation tracking (simple sum for now)
        self.prev_5day_precip = np.maximum(
            0.0,
            self.prev_5day_precip + precipitation
        )
        
        # Determine appropriate curve number based on conditions
        for i in range(self.domain_size):
            # Different thresholds for growing vs dormant season
            if growing_season[i]:
                if self.prev_5day_precip[i] < 1.4:
                    self.cn_current[i] = self.cn_dry[i]
                elif self.prev_5day_precip[i] > 2.1:
                    self.cn_current[i] = self.cn_wet[i]
                else:
                    self.cn_current[i] = self.cn_normal[i]
            else:
                if self.prev_5day_precip[i] < 0.5:
                    self.cn_current[i] = self.cn_dry[i]
                elif self.prev_5day_precip[i] > 1.1:
                    self.cn_current[i] = self.cn_wet[i]
                else:
                    self.cn_current[i] = self.cn_normal[i]
                    
    def calculate_curve_number_runoff(self, precipitation: NDArray[np.float32]) -> None:
        """Calculate runoff using SCS curve number method
        
        Args:
            precipitation: Precipitation depth for each cell
        """
        if self.landuse_indices is None:
            raise RuntimeError("Module must be initialized before calculations")
            
        for landuse_id, params in self.params.items():
            mask = (self.landuse_indices == landuse_id)
            if not np.any(mask):
                continue
                
            # Calculate potential retention
            S = (1000.0 / self.cn_current[mask] - 10.0)
            
            # Calculate initial abstraction
            Ia = params.initial_abstraction_ratio * S
            
            # Calculate runoff
            effective_precip = np.maximum(0.0, precipitation[mask] - Ia)
            self.runoff[mask] = np.where(
                effective_precip > 0.0,
                effective_precip ** 2 / (effective_precip + S),
                0.0
            )
            
            # Apply minimum threshold
            self.runoff[mask] = np.where(
                self.runoff[mask] < params.minimum_runoff,
                0.0,
                self.runoff[mask]
            )
            
            # Calculate storm drain capture
            self.storm_drain_capture[mask] = (
                self.runoff[mask] * params.storm_drain_capture_fraction
            )
            
            # Reduce runoff by storm drain capture
            self.runoff[mask] -= self.storm_drain_capture[mask]
            
    def calculate_gridded_runoff(self, date: datetime, 
                               precipitation: NDArray[np.float32]) -> None:
        """Calculate runoff using gridded method
        
        Args:
            date: Current simulation date
            precipitation: Precipitation depth for each cell
        """
        if self.runoff_zones is None or self.monthly_ratios is None:
            raise RuntimeError("Gridded parameters must be initialized")
            
        # Update ratios on first day of month
        if date.day == 1:
            month_idx = date.month - 1
            for zone_id, ratios in self.monthly_ratios.items():
                mask = (self.runoff_zones == zone_id)
                self.runoff_ratios[mask] = ratios[month_idx]
                
        # Calculate base runoff using curve number method
        self.calculate_curve_number_runoff(precipitation)
        
        # Apply ratios
        self.runoff *= self.runoff_ratios
        
    def process_timestep(self, date: datetime, precipitation: NDArray[np.float32],
                        growing_season: NDArray[np.bool_]) -> None:
        """Process all runoff calculations for current timestep
        
        Args:
            date: Current simulation date
            precipitation: Precipitation depth for each cell
            growing_season: Boolean array indicating growing season status
        """
        # Update antecedent conditions
        self.update_antecedent_conditions(precipitation, growing_season)
        
        # Calculate runoff using selected method
        if self.method == "curve_number":
            self.calculate_curve_number_runoff(precipitation)
        else:
            self.calculate_gridded_runoff(date, precipitation)
