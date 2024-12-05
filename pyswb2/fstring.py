from dataclasses import dataclass
from typing import List, Union, Optional
import re

@dataclass
class StringList:
    """Python implementation of FSTRING_LIST_T"""
    values: List[str] = None
    
    def __post_init__(self):
        self.values = self.values or []
        self.count = len(self.values)
        self.missing_value_count = sum(1 for x in self.values if not x.strip())
        
    def append(self, value: Union[str, int, float, bool]) -> None:
        """Add value to string list"""
        self.values.append(str(value))
        self.count += 1
        if not str(value).strip():
            self.missing_value_count += 1
            
    def get(self, index: int) -> str:
        """Get value at index"""
        try:
            return self.values[index-1]  # Convert to 0-based indexing
        except IndexError:
            return "<NA>"
            
    def get_integer(self) -> List[int]:
        """Get values as integers"""
        result = []
        for val in self.values:
            try:
                # Handle float strings by converting to float first
                if '.' in val:
                    result.append(int(float(val)))
                else:
                    result.append(int(val))
            except (ValueError, TypeError):
                result.append(-9999)  # NA_INT equivalent
        return result
    
    def get_float(self) -> List[float]:
        """Get values as floats"""
        result = []
        for val in self.values:
            try:
                result.append(float(val))
            except (ValueError, TypeError):
                result.append(float('-inf'))  # NA_FLOAT equivalent
        return result
        
    def get_logical(self) -> List[bool]:
        """Get values as booleans"""
        true_values = {'true', 't', 'yes', 'y', '1'}
        return [str(val).lower() in true_values for val in self.values]
        
    def clear(self) -> None:
        """Clear all values"""
        self.values = []
        self.count = 0
        self.missing_value_count = 0
        
    def sort(self, reverse: bool = False) -> None:
        """Sort values alphabetically"""
        self.values.sort(reverse=reverse)
        
    def sort_integer(self, reverse: bool = False) -> None:
        """Sort by integer values"""
        try:
            self.values.sort(key=lambda x: int(float(x)) if x.strip() else float('inf'), 
                           reverse=reverse)
        except (ValueError, TypeError):
            pass
            
    def sort_float(self, reverse: bool = False) -> None:
        """Sort by float values"""
        try:
            self.values.sort(key=lambda x: float(x) if x.strip() else float('inf'), 
                           reverse=reverse)
        except (ValueError, TypeError):
            pass
            
    def grep(self, pattern: str, ignore_case: bool = True) -> 'StringList':
        """Find values matching pattern"""
        flags = re.IGNORECASE if ignore_case else 0
        matches = [v for v in self.values if re.search(pattern, v, flags)]
        return StringList(matches)
        
    def unique(self) -> 'StringList':
        """Get unique values"""
        return StringList(list(dict.fromkeys(self.values)))

def split(text: str, delimiter: str = ',') -> StringList:
    """Split string into StringList"""
    if not text.strip():
        return StringList()
    return StringList([x.strip() for x in text.split(delimiter)])

# Constants
TAB = '\t'
WHITESPACE = ' \t'
BACKSLASH = '\\'
FORWARDSLASH = '/'
CARRIAGE_RETURN = '\r'
COMMENT_CHARACTERS = '#!%'
DOUBLE_QUOTE = '"'
PUNCTUATION = ',;:'
