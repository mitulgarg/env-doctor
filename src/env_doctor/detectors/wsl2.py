import os
import subprocess
import platform

from env_doctor.core import Detector, DetectionResult, Status, DetectorRegistry


@DetectorRegistry.register("wsl2")
class WSL2Detector(Detector):
    """
    Detects Windows Subsystem for Linux 2 (WSL2) environment and validates GPU forwarding.
    
    This detector identifies whether the current environment is:
    - Native Linux
    - WSL1 (no CUDA support available)  
    - WSL2 (GPU forwarding support available)
    
    For WSL2 environments, it validates proper GPU forwarding setup including:
    - Absence of internal NVIDIA drivers (which break GPU forwarding)
    - Presence of WSL2 CUDA libraries
    - Functional nvidia-smi command
    
    Note: WSL1 does not support CUDA at all, so this detector will warn users
    to upgrade to WSL2 for any GPU computing needs.
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
    
    def _detect_wsl2_environment(self) -> str:
        """Detect the type of WSL environment, focusing on WSL2 support."""
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
    
    def _check_wsl2_libcuda(self) -> bool:
        """Check if WSL2 CUDA library exists."""
        return os.path.exists("/usr/lib/wsl/lib/libcuda.so")
    
    def _check_internal_nvidia_driver(self) -> bool:
        """Check if internal NVIDIA driver is installed in WSL."""
        return os.path.exists("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so")
    
    def detect(self) -> DetectionResult:
        """Detect WSL2 environment and GPU forwarding status."""
        env_type = self._detect_wsl2_environment()
        result = DetectionResult(component="wsl2", status=Status.SUCCESS)
        result.version = env_type
        
        # Native Linux path
        if env_type == "native_linux":
            result.metadata["environment"] = "Native Linux"
            return result
        
        # WSL1 path - no CUDA support available
        if env_type == "wsl1":
            result.metadata["environment"] = "WSL1"
            result.issues.append("WSL1 detected. CUDA is not supported in WSL1 at all.")
            result.recommendations.append("Upgrade to WSL2 for GPU/CUDA support")
            result.status = Status.ERROR
            return result
        
        # WSL2 path - check GPU forwarding setup
        if env_type == "wsl2":
            result.metadata["environment"] = "WSL2"
            
            # Check for problematic internal NVIDIA driver
            has_internal_driver = self._check_internal_nvidia_driver()
            if has_internal_driver:
                result.status = Status.ERROR
                result.issues.append("NVIDIA driver installed inside WSL. This breaks GPU forwarding.")
                result.recommendations.append("Run: sudo apt remove --purge nvidia-*")
                return result
            
            # Check for WSL2 CUDA library
            has_libcuda = self._check_wsl2_libcuda()
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