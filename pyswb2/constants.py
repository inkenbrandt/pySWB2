
import math
from typing import Final


# Delimiters
TAB: Final = '\t'
WHITESPACE: Final = ' \t'
BACKSLASH: Final = '\\'
FORWARDSLASH: Final = '/'
CARRIAGE_RETURN: Final = '\r'
COMMENT_CHARACTERS: Final = '#!%'
DOUBLE_QUOTE: Final = '"'
PUNCTUATION: Final = ',;:'

# Default formats
DEFAULT_DATE_FORMAT: Final = 'MM/DD/YYYY'
DEFAULT_TIME_FORMAT: Final = 'HH:MM:SS'

# Constants
PI = math.pi
TWOPI = 2 * PI
HALFPI = PI / 2

# Data type identifiers (optional usage in Python)
DATATYPE_INT = 0
DATATYPE_FLOAT = 1
DATATYPE_REAL = 1
DATATYPE_DOUBLE = 2
DATATYPE_SHORT = 3
DATATYPE_NA = -9999

# Special values
NA_INT: Final = -999999
NA_FLOAT: Final = float(-999999)


# Conversion Utilities
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return celsius * 9.0 / 5.0 + 32.0

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) * 5.0 / 9.0

def meters_to_feet(meters):
    """Convert meters to feet."""
    return meters * 3.28084

def feet_to_meters(feet):
    """Convert feet to meters."""
    return feet / 3.28084

def inches_to_millimeters(inches):
    """Convert inches to millimeters."""
    return inches * 25.4

def millimeters_to_inches(mm):
    """Convert millimeters to inches."""
    return mm / 25.4
