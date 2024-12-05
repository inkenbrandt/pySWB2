from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

@dataclass
class StatisticsConfig:
    """Configuration parameters for statistics calculations"""
    output_dir: Path
    prefix: str = "stats"
    calculate_water_balance: bool = True
    calculate_landuse_summary: bool = True
    calculate_soil_summary: bool = True
    calculate_temporal_stats: bool = True
    temporal_aggregation: str = "monthly"  # 'daily', 'monthly', 'annual'

class StatisticsModule:
    """Module for calculating and managing SWB model statistics"""
    
    def __init__(self, domain_size: int, grid_shape: tuple):
        """Initialize statistics module
        
        Args:
            domain_size: Number of cells in model domain
            grid_shape: Shape of the model grid (rows, cols)
        """
        self.domain_size = domain_size
        self.grid_shape = grid_shape
        
        # Initialize tracking arrays
        self.landuse_codes = []
        self.soil_types = []
        self.poly_ids = []
        
        # Initialize statistics arrays
        self.temporal_stats = {
            'precipitation': np.zeros(domain_size, dtype=np.float32),
            'actual_et': np.zeros(domain_size, dtype=np.float32),
            'runoff': np.zeros(domain_size, dtype=np.float32),
            'infiltration': np.zeros(domain_size, dtype=np.float32),
            'soil_moisture': np.zeros(domain_size, dtype=np.float32)
        }
        
        self.cumulative_stats = {
            'total_precip': np.zeros(domain_size, dtype=np.float32),
            'total_et': np.zeros(domain_size, dtype=np.float32),
            'total_runoff': np.zeros(domain_size, dtype=np.float32),
            'total_infiltration': np.zeros(domain_size, dtype=np.float32)
        }
        
        # Tracking variables
        self.current_period = None
        self.period_count = 0
        self.version = "3.0.0"
        
    def initialize(self, landuse_indices: NDArray[np.int32],
                  soil_indices: NDArray[np.int32],
                  config: StatisticsConfig) -> None:
        """Initialize module with model data
        
        Args:
            landuse_indices: Array mapping cells to landuse types
            soil_indices: Array mapping cells to soil types
            config: Statistics configuration object
        """
        self.config = config
        self.landuse_indices = landuse_indices
        self.soil_indices = soil_indices
        
        # Create unique lists
        self.landuse_codes = list(np.unique(landuse_indices))
        self.soil_types = list(np.unique(soil_indices))
        
        # Create output directory if needed
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def update_water_balance(self, date: datetime,
                           precipitation: NDArray[np.float32],
                           actual_et: NDArray[np.float32],
                           runoff: NDArray[np.float32],
                           infiltration: NDArray[np.float32],
                           soil_moisture: NDArray[np.float32]) -> None:
        """Update water balance statistics
        
        Args:
            date: Current simulation date
            precipitation: Precipitation array
            actual_et: Actual ET array
            runoff: Runoff array
            infiltration: Infiltration array
            soil_moisture: Soil moisture array
        """
        if not self.config.calculate_water_balance:
            return
            
        # Update period tracking
        new_period = self._get_period(date)
        if new_period != self.current_period:
            self._write_period_stats()
            self._reset_temporal_stats()
            self.current_period = new_period
            self.period_count += 1
            
        # Update temporal statistics
        self.temporal_stats['precipitation'] += precipitation
        self.temporal_stats['actual_et'] += actual_et
        self.temporal_stats['runoff'] += runoff
        self.temporal_stats['infiltration'] += infiltration
        self.temporal_stats['soil_moisture'] += soil_moisture
        
        # Update cumulative statistics
        self.cumulative_stats['total_precip'] += precipitation
        self.cumulative_stats['total_et'] += actual_et
        self.cumulative_stats['total_runoff'] += runoff
        self.cumulative_stats['total_infiltration'] += infiltration
        
    def calculate_landuse_summary(self) -> Dict:
        """Calculate summary statistics by land use type
        
        Returns:
            Dictionary of land use statistics
        """
        if not self.config.calculate_landuse_summary:
            return {}
            
        summary = {
            'landuse_count': len(self.landuse_codes),
            'cells_per_landuse': {},
            'mean_stats_per_landuse': {}
        }
        
        for code in self.landuse_codes:
            mask = self.landuse_indices == code
            summary['cells_per_landuse'][code] = np.sum(mask)
            
            # Calculate mean statistics for each land use
            summary['mean_stats_per_landuse'][code] = {
                'precipitation': np.mean(self.cumulative_stats['total_precip'][mask]),
                'et': np.mean(self.cumulative_stats['total_et'][mask]),
                'runoff': np.mean(self.cumulative_stats['total_runoff'][mask]),
                'infiltration': np.mean(self.cumulative_stats['total_infiltration'][mask])
            }
            
        return summary
        
    def calculate_soil_summary(self) -> Dict:
        """Calculate summary statistics by soil type
        
        Returns:
            Dictionary of soil statistics
        """
        if not self.config.calculate_soil_summary:
            return {}
            
        summary = {
            'soil_type_count': len(self.soil_types),
            'cells_per_soil': {},
            'mean_stats_per_soil': {}
        }
        
        for soil_type in self.soil_types:
            mask = self.soil_indices == soil_type
            summary['cells_per_soil'][soil_type] = np.sum(mask)
            
            # Calculate mean statistics for each soil type
            summary['mean_stats_per_soil'][soil_type] = {
                'precipitation': np.mean(self.cumulative_stats['total_precip'][mask]),
                'et': np.mean(self.cumulative_stats['total_et'][mask]),
                'runoff': np.mean(self.cumulative_stats['total_runoff'][mask]),
                'infiltration': np.mean(self.cumulative_stats['total_infiltration'][mask])
            }
            
        return summary
        
    def _get_period(self, date: datetime) -> str:
        """Get current period identifier based on aggregation setting"""
        if self.config.temporal_aggregation == 'daily':
            return date.strftime('%Y%m%d')
        elif self.config.temporal_aggregation == 'monthly':
            return date.strftime('%Y%m')
        else:  # annual
            return date.strftime('%Y')
            
    def _write_period_stats(self) -> None:
        """Write statistics for current period"""
        if self.current_period is None:
            return
            
        # Calculate means for the period
        period_means = {
            var: values / self.period_count
            for var, values in self.temporal_stats.items()
        }
        
        # Write to file
        output_file = (self.config.output_dir / 
                      f"{self.config.prefix}_{self.current_period}.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"Statistics for period: {self.current_period}\n")
            f.write("-" * 50 + "\n")
            
            for var, values in period_means.items():
                f.write(f"{var}:\n")
                f.write(f"  Mean: {np.mean(values):.3f}\n")
                f.write(f"  Min: {np.min(values):.3f}\n")
                f.write(f"  Max: {np.max(values):.3f}\n")
                f.write(f"  StdDev: {np.std(values):.3f}\n")
                f.write("\n")
                
    def _reset_temporal_stats(self) -> None:
        """Reset temporal statistics arrays"""
        for arr in self.temporal_stats.values():
            arr.fill(0.0)
            
        self.period_count = 0
        
    def write_final_summary(self) -> None:
        """Write final summary statistics"""
        # Calculate landuse and soil summaries
        landuse_summary = self.calculate_landuse_summary()
        soil_summary = self.calculate_soil_summary()
        
        # Write summary file
        summary_file = self.config.output_dir / f"{self.config.prefix}_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("SWB Model Statistics Summary\n")
            f.write(f"Version: {self.version}\n")
            f.write("-" * 50 + "\n\n")
            
            # Write landuse summary
            if landuse_summary:
                f.write("Land Use Summary\n")
                f.write("-" * 20 + "\n")
                for code, stats in landuse_summary['mean_stats_per_landuse'].items():
                    f.write(f"\nLand Use Code: {code}\n")
                    f.write(f"Cell Count: {landuse_summary['cells_per_landuse'][code]}\n")
                    for var, val in stats.items():
                        f.write(f"{var}: {val:.3f}\n")
                        
            # Write soil summary
            if soil_summary:
                f.write("\nSoil Type Summary\n")
                f.write("-" * 20 + "\n")
                for soil_type, stats in soil_summary['mean_stats_per_soil'].items():
                    f.write(f"\nSoil Type: {soil_type}\n")
                    f.write(f"Cell Count: {soil_summary['cells_per_soil'][soil_type]}\n")
                    for var, val in stats.items():
                        f.write(f"{var}: {val:.3f}\n")
