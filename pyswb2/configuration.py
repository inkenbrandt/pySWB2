from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List
from enum import Enum
import yaml
import json
import re


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
    """Configuration for input data files"""
    landuse_grid: Path
    hydrologic_soils_group: Path  # This is the correct field name
    awc_grid: Optional[Path] = None
    weather_dir: Optional[Path] = None
    fragments_file: Optional[Path] = None
    lookup_tables: Dict[str, Path] = field(default_factory=dict)

    def validate(self):
        """Validate input configuration"""
        required = [self.landuse_grid, self.hydrologic_soils_group]
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
    write_daily: bool = True

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
    use_crop_coefficients: bool = False
    dynamic_rooting: bool = False
    routing_enabled: bool = False
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
        """Load from control file with improved parsing"""
        config = {}
        active_section = None
        output_vars = []
        debug_lines = []  # For debugging

        with open(path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Store original line for debugging
                original_line = line.strip()
                if original_line:
                    debug_lines.append(f"Line {line_num}: {original_line}")

                # Skip pure comment lines and section headers
                if any(line.lstrip().startswith(c) for c in ['!', '#', '$', '%', '*', '-', '(']):
                    continue

                # Remove trailing comments - be careful with numbers containing decimals
                comment_chars = ['!', '#', '$', '%', '*']
                min_comment_pos = len(line)
                for char in comment_chars:
                    pos = line.find(char)
                    if pos >= 0:
                        min_comment_pos = min(min_comment_pos, pos)
                line = line[:min_comment_pos].strip()

                if not line:
                    continue

                try:
                    # Handle OUTPUT ENABLE lines specially
                    if line.upper().startswith('OUTPUT ENABLE'):
                        var = line.split('OUTPUT ENABLE', 1)[1].strip()
                        output_vars.append(var)
                        continue

                    # Split the line into words
                    parts = line.split()
                    if not parts:
                        continue

                    # Get the key (first word) and combine remaining parts as value
                    key = parts[0].lower()  # Convert key to lowercase
                    if len(parts) > 1:
                        value = ' '.join(parts[1:])
                        if key == 'grid' and not any(c in value for c in comment_chars):
                            print(f"Found valid GRID configuration: {value}")
                        config[key] = value.strip()
                        debug_lines.append(f"Stored config: {key} = {value.strip()}")

                except Exception as e:
                    print(f"Warning: Error parsing line {line_num}: '{line}' - {str(e)}")
                    continue

        # Debug output
        print("\nParsed control file contents:")
        print("\n".join(debug_lines))
        print("\nFound configuration keys:", sorted(list(config.keys())))

        # Add collected output variables
        if output_vars:
            config['output_variables'] = output_vars

        return cls._parse_dict(config)

    @classmethod
    def _parse_dict(cls, config: Dict) -> 'ModelConfig':
        """Parse configuration dictionary with improved error handling"""
        # Parse grid configuration
        grid_value = None

        # Find the grid configuration
        for key in config:
            if key.lower() == 'grid':
                value = config[key].strip()
                # Verify this looks like a valid grid configuration
                parts = value.split()
                if len(parts) >= 5 and all(part.replace('.', '').replace('-', '').isdigit() for part in parts[:5]):
                    grid_value = value
                    print(f"Using grid configuration: {grid_value}")
                    break

        if not grid_value:
            print("\nConfiguration dump for debugging:")
            for key, value in sorted(config.items()):
                print(f"{key}: {value}")
            raise ValueError("Valid grid configuration not found in control file.")

        grid_parts = grid_value.split()
        if len(grid_parts) < 5:
            raise ValueError(f"Invalid grid configuration: {grid_value}. Expected 5 values: NX NY X0 Y0 CELL_SIZE")

        try:
            grid = GridConfig(
                nx=int(float(grid_parts[0])),
                ny=int(float(grid_parts[1])),
                x0=float(grid_parts[2].rstrip('.')),
                y0=float(grid_parts[3].rstrip('.')),
                cell_size=float(grid_parts[4].rstrip('.')),
                proj4_string=config.get('base_projection_definition')
            )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing grid values: {str(e)}\nGrid parts: {grid_parts}")

        # Parse input configuration
        try:
            def get_path_from_config(key: str, default: Optional[str] = None) -> Optional[Path]:
                if key not in config:
                    return Path(default) if default else None
                parts = config[key].split()
                return Path(parts[-1]) if parts else None

            input_config = InputConfig(
                landuse_grid=get_path_from_config('land_use', 'input/landuse.asc'),
                hydrologic_soils_group=get_path_from_config('hydrologic_soils_group', 'input/soils.asc'),
                awc_grid=get_path_from_config('soil_storage_max', None),
                weather_dir=Path(config['weather_dir']) if 'weather_dir' in config else None,
                fragments_file=Path(config['fragments_daily_file']) if 'fragments_daily_file' in config else None,
                lookup_tables={
                    'landuse': Path(config['land_use_lookup_table']) if 'land_use_lookup_table' in config else None
                }
            )
        except Exception as e:
            print("\nDebug: Input configuration parsing failed")
            print("Available configuration keys:", sorted(list(config.keys())))
            print("\nValues for key input parameters:")
            for key in ['land_use', 'hydrologic_soils_group', 'soil_storage_max',
                        'weather_dir', 'fragments_daily_file', 'land_use_lookup_table']:
                if key in config:
                    print(f"{key}: {config[key]}")
            raise ValueError(f"Error parsing input configuration: {str(e)}")

        # Parse output configuration
        try:
            output_dir = Path(config.get('output_dir', 'output'))
            output_config = OutputConfig(
                directory=output_dir,
                prefix=config.get('output_prefix', ''),
                format=OutputFormat(config.get('output_format', 'netcdf').lower()),
                variables=config.get('output_variables', []),
                compression=config.get('compress_output', True),
                write_daily=True
            )
        except Exception as e:
            raise ValueError(f"Error parsing output configuration: {str(e)}")

        # Parse dates
        try:
            start_date = datetime.strptime(config['start_date'], '%m/%d/%Y')
            end_date = datetime.strptime(config['end_date'], '%m/%d/%Y')
        except KeyError as e:
            raise ValueError(f"Missing required date configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Error parsing dates: {e}. Dates should be in MM/DD/YYYY format.")

        if start_date > end_date:
            raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")

        return cls(
            grid=grid,
            input=input_config,
            output=output_config,
            start_date=start_date,
            end_date=end_date,
            use_crop_coefficients=config.get('crop_coefficient_method', 'NONE').upper() != 'NONE',
            dynamic_rooting=True,
            routing_enabled=config.get('flow_routing_method', 'NONE').upper() != 'NONE'
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
        """Save configuration as YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False)

    def _save_json(self, path: Path) -> None:
        """Save configuration as JSON"""
        with open(path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

    def _save_control(self, path: Path) -> None:
        """Save configuration in control file format"""
        with open(path, 'w') as f:
            # Write header
            f.write("# SWB Model Configuration\n")
            f.write("# Generated by ConfigurationManager\n\n")

            # Write grid configuration
            f.write("(1) GRID DEFINITION\n")
            f.write("-" * 50 + "\n")
            f.write(f"GRID {self.config.grid.nx} {self.config.grid.ny} "
                   f"{self.config.grid.x0} {self.config.grid.y0} {self.config.grid.cell_size}\n")
            if self.config.grid.proj4_string:
                f.write(f"BASE_PROJECTION_DEFINITION {self.config.grid.proj4_string}\n")
            f.write("\n")

            # Write input configuration
            f.write("(2) INPUT FILES\n")
            f.write("-" * 50 + "\n")
            f.write(f"LAND_USE ARC_GRID {self.config.input.landuse_grid}\n")
            f.write(f"HYDROLOGIC_SOILS_GROUP ARC_GRID {self.config.input.soils_grid}\n")
            if self.config.input.awc_grid:
                f.write(f"SOIL_STORAGE_MAX ARC_GRID {self.config.input.awc_grid}\n")
            if self.config.input.fragments_file:
                f.write(f"FRAGMENTS_DAILY_FILE {self.config.input.fragments_file}\n")
            f.write("\n")

            # Write output configuration
            f.write("(3) OUTPUT CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"OUTPUT_DIR {self.config.output.directory}\n")
            f.write(f"OUTPUT_PREFIX {self.config.output.prefix}\n")
            f.write(f"OUTPUT_FORMAT {self.config.output.format.value}\n")
            for var in self.config.output.variables:
                f.write(f"OUTPUT ENABLE {var}\n")
            f.write("\n")

            # Write date configuration
            f.write("(4) SIMULATION PERIOD\n")
            f.write("-" * 50 + "\n")
            f.write(f"START_DATE {self.config.start_date.strftime('%m/%d/%Y')}\n")
            f.write(f"END_DATE {self.config.end_date.strftime('%m/%d/%Y')}\n")