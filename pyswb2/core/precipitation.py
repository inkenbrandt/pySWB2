import numpy as np
from pathlib import Path
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime

@dataclass
class FragmentSet:
    rain_gage_zone: int
    start_record: Dict[int, int]
    number_of_fragments: Dict[int, int]

@dataclass
class PrecipitationData:
    date: datetime
    gross_precip: float
    rainfall: float
    snowfall: float
    net_rainfall: float
    net_snowfall: float

class Precipitation:
    def __init__(self, grid_shape: tuple):
        # Constants
        self.FREEZING_F = 32.0
        self.NEAR_ZERO = 1.0e-9
        
        # Grid initialization
        self.grid_shape = grid_shape
        self.precipitation = np.zeros(grid_shape, dtype=np.float32)
        self.snowfall = np.zeros(grid_shape, dtype=np.float32)
        self.rainfall = np.zeros(grid_shape, dtype=np.float32)
        
        # Method of Fragments variables
        self.fragments = []
        self.fragments_sets = {}
        self.current_fragments = None
        self.random_values = None
        self.rainfall_zones = None
        
        # State tracking
        self._current_data: Optional[PrecipitationData] = None
        self.simulation_number = 1
        
    def initialize(self, fragments_file: Path, rainfall_zones: np.ndarray, 
                  random_start: int = 12345) -> None:
        """Initialize precipitation module with fragments and zones"""
        self.rainfall_zones = rainfall_zones
        np.random.seed(random_start)
        self._read_daily_fragments(fragments_file)
        self._process_fragment_sets()
        
    def _read_daily_fragments(self, filename: Path) -> None:
        """Read daily precipitation fragments from file"""
        with open(filename) as f:
            for line in f:
                if line.startswith(("#", "%", "!")):
                    continue
                    
                parts = line.split()
                month = int(parts[0])
                rain_gage_zone = int(parts[1])
                fragment_set = int(parts[2])
                fragment_values = [float(x) for x in parts[3:]]
                
                # Validate and normalize fragment values
                fragment_values = [max(0.0, min(1.0, x)) for x in fragment_values]
                if month == 2:
                    self._normalize_february_fragments(fragment_values)
                    
                self.fragments.append({
                    'month': month,
                    'zone': rain_gage_zone,
                    'set': fragment_set,
                    'values': fragment_values
                })
    
    def _process_fragment_sets(self) -> None:
        """Process fragments into organized sets by zone and month"""
        max_zone = max(f['zone'] for f in self.fragments)
        for zone in range(1, max_zone + 1):
            fragment_set = FragmentSet(
                rain_gage_zone=zone,
                start_record={m: 0 for m in range(1, 13)},
                number_of_fragments={m: 0 for m in range(1, 13)}
            )
            
            # Find fragments for each month in this zone
            for month in range(1, 13):
                fragments = [i for i, f in enumerate(self.fragments) 
                           if f['zone'] == zone and f['month'] == month]
                if fragments:
                    fragment_set.start_record[month] = min(fragments)
                    fragment_set.number_of_fragments[month] = len(fragments)
                    
            self.fragments_sets[zone] = fragment_set
    
    def calculate_precipitation(self, date: datetime, tmin: float, tmax: float, 
                              interception: float) -> PrecipitationData:
        """Calculate precipitation components for current timestep"""
        # Get fragments for current date
        precip = self._get_precipitation_from_fragments(date)
        
        # Partition into rain and snow
        snowfall, net_snowfall, rainfall, net_rainfall = self._partition_precipitation(
            tmin, tmax, interception, precip)
            
        self._current_data = PrecipitationData(
            date=date,
            gross_precip=precip,
            rainfall=rainfall,
            snowfall=snowfall,
            net_rainfall=net_rainfall,
            net_snowfall=net_snowfall
        )
        
        return self._current_data
    
    def calculate_snowmelt(self, tmin: float, tmax: float) -> float:
        """Calculate potential snowmelt"""
        if tmax > self.FREEZING_F:
            melt_temp = max(0.0, (tmin + tmax) / 2 - self.FREEZING_F)
            return 1.5 * melt_temp
        return 0.0
    
    def _partition_precipitation(self, tmin: float, tmax: float, 
                               interception: float, gross_precip: float) -> Tuple[float, float, float, float]:
        """Partition precipitation into rain and snow based on temperature"""
        if tmax <= self.FREEZING_F:
            snowfall = gross_precip
            rainfall = 0.0
        elif tmin >= self.FREEZING_F:
            snowfall = 0.0
            rainfall = gross_precip
        else:
            fraction_snow = (self.FREEZING_F - tmin) / (tmax - tmin)
            snowfall = gross_precip * fraction_snow
            rainfall = gross_precip - snowfall
            
        net_snowfall = max(0.0, snowfall - interception)
        net_rainfall = max(0.0, rainfall - interception)
        
        return snowfall, net_snowfall, rainfall, net_rainfall
    
    def _get_precipitation_from_fragments(self, date: datetime) -> float:
        """Get precipitation value from fragments for current date"""
        month = date.month
        day = date.day
        
        if day == 1:
            self._update_fragments(True, month)
        else:
            self._update_fragments(False, month)
            
        precip = 0.0
        for zone in np.unique(self.rainfall_zones):
            if zone > 0:  # Skip no-data zones
                fragment = self._get_current_fragment(zone, month)
                if fragment:
                    precip += fragment['values'][day - 1]
                    
        return precip
    
    def _update_fragments(self, shuffle: bool, month: int) -> None:
        """Update fragment selection for current timestep"""
        if shuffle:
            self.random_values = np.random.random(self.random_values.shape)
            
        for zone in np.unique(self.rainfall_zones):
            if zone > 0:
                fragment_set = self.fragments_sets.get(zone)
                if fragment_set:
                    start = fragment_set.start_record[month]
                    count = fragment_set.number_of_fragments[month]
                    if count > 0:
                        idx = start + int(self.random_values[zone - 1] * count)
                        self.current_fragments[zone - 1] = self.fragments[idx]
    
    def _normalize_february_fragments(self, values: list) -> None:
        """Normalize February fragment values"""
        if values[28] > 0:  # Has day 29
            sum_28 = sum(values[:28])
            if sum_28 > 0:
                values[:28] = [v / sum_28 for v in values[:28]]
            values[28:] = [0.0] * (len(values) - 28)
    
    def _get_current_fragment(self, zone: int, month: int) -> Optional[Dict]:
        """Get current fragment for given zone and month"""
        if self.current_fragments is None or zone <= 0:
            return None
        return self.current_fragments[zone - 1]
