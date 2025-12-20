"""
Registry system for Env-Doctor plugins.

Enables automatic discovery and registration of Detector classes.

Example:
    # Register a detector
    @DetectorRegistry.register("my_detector")
    class MyDetector(Detector):
        ...
    
    # Use registry to get all detectors
    detectors = DetectorRegistry.all()
    for detector in detectors:
        result = detector.detect()
"""
from typing import Dict, List, Type
from .detector import Detector
from .exceptions import DetectorRegistrationError, DetectorNotFoundError

class DetectorRegistry:
    """
    Central registry for all environment detectors.
    """
    _detectors: Dict[str, Type[Detector]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a class as a detector.
        
        Args:
            name: Unique identifier for the detector.
            
        Returns:
            The decorator function.
            
        Raises:
            DetectorRegistrationError: If name is already registered.
        """
        def decorator(detector_class: Type[Detector]):
            if name in cls._detectors:
                raise DetectorRegistrationError(f"Detector '{name}' already registered.")
            cls._detectors[name] = detector_class
            return detector_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Detector:
        """
        Get an instance of a registered detector by name.
        
        Args:
            name: The unique name of the detector.
            
        Returns:
            Detector: An instantiated detector.
            
        Raises:
            DetectorNotFoundError: If detector is not found.
        """
        if name not in cls._detectors:
            raise DetectorNotFoundError(f"Detector '{name}' not found.")
        return cls._detectors[name]()

    @classmethod
    def all(cls) -> List[Detector]:
        """
        Get instances of all registered detectors.
        
        Returns:
            List[Detector]: List of instantiated detectors.
        """
        return [klass() for klass in cls._detectors.values()]

    @classmethod
    def get_names(cls) -> List[str]:
        """
        Get list of all registered detector names.
        
        Returns:
            List[str]: Sorted list of detector names.
        """
        return sorted(list(cls._detectors.keys()))
