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


class InstallerError(Exception):
    """Base exception for installer-related errors."""
    pass


class PrivilegeError(InstallerError):
    """Raised when insufficient privileges for installation."""
    pass


class StepExecutionError(InstallerError):
    """Raised when an installation step fails."""
    def __init__(self, step_index, command, return_code, stderr):
        self.step_index = step_index
        self.command = command
        self.return_code = return_code
        self.stderr = stderr
        super().__init__(
            f"Step {step_index} failed (exit code {return_code}): {command}"
        )
