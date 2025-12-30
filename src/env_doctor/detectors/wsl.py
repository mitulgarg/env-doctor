import os
import subprocess
import platform

from env_doctor.core import Detector, DetectionResult, Status, DetectorRegistry


@DetectorRegistry.register("wsl")
class WSLDetector(Detector):
    """
    Detects Windows Subsystem for Linux (WSL) environment and validates GPU forwarding.
    
    This detector identifies whether the current environment is:
    - Native Linux
    - WSL1 (limited GPU support)  
    - WSL2 (full GPU forwarding support)
    
    For WSL2 environments, it validates proper GPU forwarding setup including:
    - Absence of internal NVIDIA drivers
    - Presence of WSL CUDA libraries
    - Functional nvidia-smi command
    """
    
    def can_run(self) -> bool:
        """Check if this detector can run on the current platform."""
        return platform.system() == "Linux"
    
    def _read_proc_version(self) -> str:
        """Read /proc/version file to determine kernel version."""
        try:
            with open("/proc/version", "r") as f:
                return f.read().strip()
        except Exception:
            return ""
    
    def _detect_wsl_type(self) -> str:
        """Detect the type of WSL environment."""
        version_info = self._read_proc_version()
        
        if not version_info:
            return "native_linux"
        
        version_lower = version_info.lower()
        if "microsoft" in version_lower:
            if "wsl2" in version_lower:
                return "wsl2"
            else:
                return "wsl1"
        
        return "native_linux"
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi command works."""
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_wsl_libcuda(self) -> bool:
        """Check if WSL CUDA library exists."""
        return os.path.exists("/usr/lib/wsl/lib/libcuda.so")
    
    def _check_internal_nvidia_driver(self) -> bool:
        """Check if internal NVIDIA driver is installed in WSL."""
        return os.path.exists("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so")
    
    def detect(self) -> DetectionResult:
        """Detect WSL environment and GPU forwarding status."""
        wsl_type = self._detect_wsl_type()
        result = DetectionResult(component="wsl", status=Status.SUCCESS)
        result.version = wsl_type
        
        # Native Linux path
        if wsl_type == "native_linux":
            result.metadata["environment"] = "Native Linux"
            return result
        
        # WSL1 path
        if wsl_type == "wsl1":
            result.metadata["environment"] = "WSL1"
            result.issues.append("WSL1 detected. GPU passthrough not supported in WSL1.")
            result.recommendations.append("Upgrade to WSL2 for GPU support")
            result.status = Status.WARNING
            return result
        
        # WSL2 path - check GPU forwarding setup
        if wsl_type == "wsl2":
            result.metadata["environment"] = "WSL2"
            
            # Check for problematic internal NVIDIA driver
            has_internal_driver = self._check_internal_nvidia_driver()
            if has_internal_driver:
                result.status = Status.ERROR
                result.issues.append("NVIDIA driver installed inside WSL. This breaks GPU forwarding.")
                result.recommendations.append("Run: sudo apt remove --purge nvidia-*")
                return result
            
            # Check for WSL CUDA library
            has_libcuda = self._check_wsl_libcuda()
            if not has_libcuda:
                result.status = Status.ERROR
                result.issues.append("Missing /usr/lib/wsl/lib/libcuda.so")
                result.recommendations.append("Reinstall NVIDIA driver on Windows host")
                return result
            
            # Check nvidia-smi functionality
            nvidia_smi_works = self._check_nvidia_smi()
            if not nvidia_smi_works:
                result.status = Status.ERROR
                result.issues.append("nvidia-smi command failed")
                result.recommendations.append("Install NVIDIA driver on Windows (version 470.76 or newer)")
                return result
            
            # All checks passed
            result.metadata["gpu_forwarding"] = "enabled"
            return result
        
        return result