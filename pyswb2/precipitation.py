import numpy as np
from pathlib import Path
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime
from numpy.typing import NDArray

@dataclass
class FragmentSet:
    rain_gage_zone: int
    start_record: Dict[int, int]
    number_of_fragments: Dict[int, int]

@dataclass
class PrecipitationData:
    """Container for precipitation data using arrays"""
    date: datetime
    gross_precip: NDArray[np.float32]
    rainfall: NDArray[np.float32]
    snowfall: NDArray[np.float32]
    net_rainfall: NDArray[np.float32]
    net_snowfall: NDArray[np.float32]


class Precipitation:
    def __init__(self, grid_shape: tuple):
        # Previous initialization code remains the same...
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
        self.rainfall_zones = None

        # Initialize random_values with a default shape to match max possible zones
        # This will be resized in initialize() when actual zones are known
        self.random_values = np.random.random(100)  # Default size, will be resized

        # State tracking
        self._current_data = None
        self.simulation_number = 1

    def initialize(self, fragments_file: Path, rainfall_zones: np.ndarray,
                   random_start: int = 12345) -> None:
        """Initialize precipitation module with fragments and zones"""
        self.rainfall_zones = rainfall_zones

        # Get number of unique zones and resize random values array
        max_zone = int(np.max(rainfall_zones))
        self.random_values = np.random.random(max_zone)

        # Set random seed
        np.random.seed(random_start)

        # Initialize fragments arrays
        self._read_daily_fragments(fragments_file)
        self._process_fragment_sets()

        # Initialize current fragments array to match number of zones
        self.current_fragments = [None] * max_zone

    def _read_daily_fragments(self, filename: Path) -> None:
        """Read daily precipitation fragments from file"""
        if not filename.exists():
            raise FileNotFoundError(f"Fragments file not found: {filename}")

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

    def calculate_snowmelt(self, tmin: float, tmax: float) -> float:
        """Calculate potential snowmelt"""
        if tmax > self.FREEZING_F:
            melt_temp = max(0.0, (tmin + tmax) / 2 - self.FREEZING_F)
            return 1.5 * melt_temp
        return 0.0

    def _partition_precipitation(self, tmin: NDArray[np.float32],
                                 tmax: NDArray[np.float32],
                                 interception: NDArray[np.float32],
                                 gross_precip: NDArray[np.float32]) -> Tuple[NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32]]:
        """Partition precipitation into rain and snow based on temperature

        Args:
            tmin: Array of minimum temperatures
            tmax: Array of maximum temperatures
            interception: Array of interception values
            gross_precip: Array of gross precipitation values

        Returns:
            Tuple of (snowfall, net_snowfall, rainfall, net_rainfall) arrays
        """
        # Initialize output arrays
        snowfall = np.zeros_like(gross_precip)
        rainfall = np.zeros_like(gross_precip)

        # All snow when max temp is below freezing
        snow_mask = tmax <= self.FREEZING_F
        snowfall[snow_mask] = gross_precip[snow_mask]

        # All rain when min temp is above freezing
        rain_mask = tmin >= self.FREEZING_F
        rainfall[rain_mask] = gross_precip[rain_mask]

        # Mixed precipitation for temps straddling freezing point
        mixed_mask = ~snow_mask & ~rain_mask
        if np.any(mixed_mask):
            fraction_snow = ((self.FREEZING_F - tmin[mixed_mask]) /
                             (tmax[mixed_mask] - tmin[mixed_mask]))
            snowfall[mixed_mask] = gross_precip[mixed_mask] * fraction_snow
            rainfall[mixed_mask] = gross_precip[mixed_mask] * (1 - fraction_snow)

        # Calculate net values after interception
        net_snowfall = np.maximum(0.0, snowfall - interception)
        net_rainfall = np.maximum(0.0, rainfall - interception)

        return snowfall, net_snowfall, rainfall, net_rainfall

    def calculate_precipitation(self, date: datetime,
                                tmin: NDArray[np.float32],
                                tmax: NDArray[np.float32],
                                interception: NDArray[np.float32]) -> PrecipitationData:
        """Calculate precipitation components for current timestep"""
        # Get fragments for current date
        precip = self._get_precipitation_from_fragments(date)

        # Ensure precip is broadcast to match temperature arrays
        if np.isscalar(precip):
            precip = np.full_like(tmin, precip)

        # Partition into rain and snow
        snowfall, net_snowfall, rainfall, net_rainfall = self._partition_precipitation(
            tmin, tmax, interception, precip)

        return PrecipitationData(
            date=date,
            gross_precip=precip,
            rainfall=rainfall,
            snowfall=snowfall,
            net_rainfall=net_rainfall,
            net_snowfall=net_snowfall
        )

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
