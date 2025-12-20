"""
Core interfaces for the Env-Doctor detection system.

This module defines the base Detector class and result types used by all
environment capability detectors.

Example:
    @DetectorRegistry.register("driver")
    class DriverDetector(Detector):
        def detect(self) -> DetectionResult:
            # Detection logic here
            return DetectionResult(
                component="driver",
                status=Status.SUCCESS
            )
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class Status(Enum):
    """Status of a detection check."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    NOT_FOUND = "not_found"

@dataclass
class DetectionResult:
    """
    Result returned by a detector.
    
    Attributes:
        component: Name of the component being checked (e.g. "nvidia_driver")
        status: Status outcome of the check
        version: Detected version string (optional)
        path: Path to the detected component (optional)
        metadata: Additional arbitrary data about the component
        issues: List of discovered issues or error messages
        recommendations: List of suggested fixes or improvements
    """
    component: str
    status: Status
    version: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        """Returns True if the component was successfully detected and verified."""
        return self.status == Status.SUCCESS

    def add_issue(self, issue: str) -> None:
        """Add an issue description to the result."""
        self.issues.append(issue)

    def add_recommendation(self, rec: str) -> None:
        """Add a recommendation to the result."""
        self.recommendations.append(rec)

class Detector(ABC):
    """
    Abstract base class for all environment detectors.
    
    Subclasses must implement the `detect()` method and be registered
    via the DetectorRegistry.
    """

    @abstractmethod
    def detect(self) -> DetectionResult:
        """
        Perform detection of the specific component.
        
        Returns:
            DetectionResult: The outcome of the detection process.
        """
        pass

    def can_run(self) -> bool:
        """
        Check if this detector can run in the current environment.
        
        Override this if detector relies on OS-specific features (e.g. Windows only).
        
        Returns:
            bool: True if detection can be attempted, False otherwise.
        """
        return True

    @property
    def name(self) -> str:
        """
        Get the name of the detector (derived from class name).
        
        Returns:
            str: Class name with 'Detector' suffix derived and lowercased.
        """
        return self.__class__.__name__.replace("Detector", "").lower()
