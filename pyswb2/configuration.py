from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List
from enum import Enum
import yaml
import json


class OutputFormat(Enum):
    NETCDF = "netcdf"
    ASCII = "ascii"

@dataclass
class GridConfig:
    nx: int
    ny: int
    x0: float
    y0: float
    cell_size: float
    proj4_string: Optional[str] = None
    
    def validate(self):
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.cell_size <= 0:
            raise ValueError("Cell size must be positive")

@dataclass
class InputConfig:
    landuse_grid: Path
    soils_grid: Path
    awc_grid: Optional[Path] = None
    weather_dir: Optional[Path] = None
    lookup_tables: Dict[str, Path] = field(default_factory=dict)
    
    def validate(self):
        required = [self.landuse_grid, self.soils_grid]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required input file not found: {path}")

@dataclass
class OutputConfig:
    directory: Path
    prefix: str
    format: OutputFormat = OutputFormat.NETCDF
    variables: List[str] = field(default_factory=list)
    compression: bool = True
    
    def validate(self):
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

@dataclass
class ModelConfig:
    grid: GridConfig
    input: InputConfig
    output: OutputConfig
    start_date: datetime
    end_date: datetime
    parameters: Dict[str, Union[float, int, str]] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: Path) -> 'ModelConfig':
        """Load configuration from file"""
        if path.suffix in ['.yml', '.yaml']:
            return cls._from_yaml(path)
        elif path.suffix == '.json':
            return cls._from_json(path)
        else:
            return cls._from_control(path)
    
    @classmethod
    def _from_yaml(cls, path: Path) -> 'ModelConfig':
        """Load from YAML file"""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls._parse_dict(config)
    
    @classmethod
    def _from_json(cls, path: Path) -> 'ModelConfig':
        """Load from JSON file"""
        with open(path) as f:
            config = json.load(f)
        return cls._parse_dict(config)
    
    @classmethod
    def _from_control(cls, path: Path) -> 'ModelConfig':
        """Load from control file"""
        config = {}
        with open(path) as f:
            for line in f:
                if line.strip() and not line.startswith(('#', '!', '%')):
                    key, value = line.split(maxsplit=1)
                    config[key.lower()] = value.strip()
        return cls._parse_dict(config)
    
    @classmethod
    def _parse_dict(cls, config: Dict) -> 'ModelConfig':
        """Parse configuration dictionary"""
        grid_parts = config['grid'].split()
        
        return cls(
            grid=GridConfig(
                nx=int(grid_parts[0]),
                ny=int(grid_parts[1]),
                x0=float(grid_parts[2]),
                y0=float(grid_parts[3]),
                cell_size=float(grid_parts[4]),
                proj4_string=config.get('proj4_string')
            ),
            input=InputConfig(
                landuse_grid=Path(config['landuse_grid']),
                soils_grid=Path(config['soils_grid']),
                awc_grid=Path(config['awc_grid']) if 'awc_grid' in config else None,
                weather_dir=Path(config['weather_dir']) if 'weather_dir' in config else None,
                lookup_tables={k.replace('lookup_table_', ''): Path(v) 
                             for k, v in config.items() 
                             if k.startswith('lookup_table_')}
            ),
            output=OutputConfig(
                directory=Path(config.get('output_dir', '../refined_files')),
                prefix=config.get('output_prefix', ''),
                format=OutputFormat(config.get('output_format', 'netcdf')),
                variables=config.get('output_variables', []),
                compression=config.get('compress_output', True)
            ),
            start_date=datetime.strptime(config['start_date'], '%m/%d/%Y'),
            end_date=datetime.strptime(config['end_date'], '%m/%d/%Y'),
            parameters={k: v for k, v in config.items() 
                       if k not in ['grid', 'landuse_grid', 'soils_grid', 
                                  'awc_grid', 'weather_dir', 'output_dir', 
                                  'output_prefix', 'output_format', 
                                  'start_date', 'end_date']}
        )
        
    def validate(self):
        """Validate entire configuration"""
        self.grid.validate()
        self.input.validate()
        self.output.validate()
        
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")

class ConfigurationManager:
    """Manages model configuration"""
    
    def __init__(self):
        self.config: Optional[ModelConfig] = None
        
    def load(self, path: Path) -> ModelConfig:
        """Load and validate configuration"""
        self.config = ModelConfig.from_file(path)
        self.config.validate()
        return self.config
    
    def save(self, path: Path) -> None:
        """Save current configuration"""
        if not self.config:
            raise RuntimeError("No configuration loaded")
            
        if path.suffix in ['.yml', '.yaml']:
            self._save_yaml(path)
        elif path.suffix == '.json':
            self._save_json(path)
        else:
            self._save_control(path)
    
    def _save_yaml(self, path: Path) -> None:
        with open(path, 'w') as f:
            yaml.dump(self.config.__dict__, f)
    
    def _save_json(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def _save_control(self, path: Path) -> None:
        with open(path, 'w') as f:
            for key, value in self.config.__dict__.items():
                f.write(f"{key.upper()} {value}\n")
