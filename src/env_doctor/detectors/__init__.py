"""
Environment detectors for env-doctor.

This module auto-imports all detectors so they register themselves.
"""
from .nvidia_driver import NvidiaDriverDetector
from .cuda_toolkit import CudaToolkitDetector
from .python_libraries import PythonLibraryDetector
from .wsl2 import WSL2Detector

__all__ = [
    'NvidiaDriverDetector',
    'CudaToolkitDetector',
    'PythonLibraryDetector',
    'WSL2Detector'
]