from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import numpy as np
from pathlib import Path
import re
from datetime import datetime

from .exceptions import ParameterError, ParameterNotFoundError, ParameterParseError
from .constants import *
from .logging_config import ParameterLogger

@dataclass 
class DictionaryEntry:
    key: str = ""
    secondary_key: str = ""
    values: List[str] = None
    
    def __post_init__(self):
        if self.values is None:
            self.values = []
            
    def add_value(self, value: Any) -> None:
        self.values.append(str(value))
        
    def get_values(self, dtype: str = "str") -> List[Any]:
        try:
            if dtype == "int":
                return [int(v) for v in self.values]
            elif dtype == "float":
                return [float(v) for v in self.values]
            elif dtype == "bool":
                return [v.lower() in ('true', 't', 'yes', 'y', '1') for v in self.values]
            return self.values
        except (ValueError, TypeError) as e:
            raise ParameterParseError(f"Could not parse values as {dtype}: {e}")

class Parameters:
    def __init__(self, log_dir: Optional[Path] = None):
        self.filenames: List[str] = []
        self.delimiters: List[str] = []
        self.comment_chars: List[str] = []
        self.count: int = 0
        self.params_dict: Dict[str, DictionaryEntry] = {}
        self.logger = ParameterLogger(log_dir)

    def add_file(self, filename: str, delimiters: str = TAB, 
                 comment_chars: str = COMMENT_CHARACTERS) -> None:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Parameter file not found: {filename}")
            
        self.filenames.append(str(path))
        self.delimiters.append(delimiters)
        self.comment_chars.append(comment_chars)
        self.count += 1
        self.logger.info(f"Added parameter file: {filename}")

    def munge_files(self) -> None:
        for i, filename in enumerate(self.filenames):
            try:
                self._process_file(filename, self.delimiters[i], self.comment_chars[i])
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
                raise

    def _process_file(self, filename: str, delimiters: str, comment_chars: str) -> None:
        with open(filename) as f:
            header = None
            line_num = 0
            
            for line in f:
                line_num += 1
                line = line.strip()
                
                if not line or line[0] in comment_chars:
                    continue
                    
                try:
                    if header is None:
                        header = [self._clean_header(col) for col in line.split(delimiters)]
                        continue

                    values = line.split(delimiters)
                    if len(values) != len(header):
                        raise ParameterParseError(
                            f"Line {line_num}: Expected {len(header)} values, got {len(values)}"
                        )

                    for col, val in zip(header, values):
                        if col not in self.params_dict:
                            self.params_dict[col] = DictionaryEntry(key=col)
                        self.params_dict[col].add_value(val.strip())
                        
                except Exception as e:
                    self.logger.error(f"Error on line {line_num}: {e}")
                    raise

    def _clean_header(self, header: str) -> str:
        """Clean and validate header column name"""
        header = re.sub(r'[^a-zA-Z0-9_]', '_', header.strip())
        if not header:
            raise ParameterParseError("Empty header column name")
        return header

    def get_parameters(self, key: str, dtype: str = "str") -> List[Any]:
        """Get parameter values for a given key"""
        if key not in self.params_dict:
            raise ParameterNotFoundError(f"Parameter key not found: {key}")
            
        return self.params_dict[key].get_values(dtype)

    def grep_name(self, pattern: str) -> List[str]:
        """Find parameter names matching pattern"""
        matches = []
        pattern = pattern.lower()
        for key in self.params_dict:
            if pattern in key.lower():
                matches.append(key)
        return matches

    def get_parameter_table(self, prefix: str, dtype: str = "float") -> np.ndarray:
        """Get multiple parameters as a 2D array"""
        keys = self.grep_name(prefix)
        if not keys:
            raise ParameterNotFoundError(f"No parameters found matching prefix: {prefix}")
            
        values = [self.get_parameters(key, dtype) for key in keys]
        return np.array(values).T

# Global instance
PARAMS = Parameters()


