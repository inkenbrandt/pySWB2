class ParameterError(Exception):
    """Base exception for parameter-related errors"""
    pass

class ParameterNotFoundError(ParameterError):
    """Exception raised when parameter key is not found"""
    pass

class ParameterParseError(ParameterError):
    """Exception raised when parameter value cannot be parsed"""
    pass
