"""
Core exceptions for the Env-Doctor detector system.
"""

class DetectorError(Exception):
    """Base exception for all detector-related errors."""
    pass

class DetectorNotFoundError(DetectorError):
    """Raised when a requested detector cannot be found in the registry."""
    pass

class DetectorRegistrationError(DetectorError):
    """Raised when there is an error registering a detector (e.g. duplicate name)."""
    pass
