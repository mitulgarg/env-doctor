"""
Environment detectors for env-doctor.

This module auto-imports all detectors so they register themselves.
"""
from .nvidia_driver import NvidiaDriverDetector
from .cuda_toolkit import CudaToolkitDetector
from .python_libraries import PythonLibraryDetector

__all__ = [
    'NvidiaDriverDetector',
    'CudaToolkitDetector',
    'PythonLibraryDetector',
]