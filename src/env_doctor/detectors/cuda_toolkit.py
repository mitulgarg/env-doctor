"""
Complete CUDA Toolkit Detector - Phase 2 Implementation

Detects CUDA toolkit installations with comprehensive checks:
- nvcc version and path
- libcudart version
- CUDA_HOME environment variable
- PATH/LD_LIBRARY_PATH correctness
- Multiple CUDA toolkit detection
- Driver compatibility validation
"""
import shutil
import subprocess
import os
import re
import platform
import glob
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from env_doctor.core.detector import Detector, DetectionResult, Status
from env_doctor.core.registry import DetectorRegistry


@DetectorRegistry.register("cuda_toolkit")
class CudaToolkitDetector(Detector):
    """
    Comprehensive CUDA Toolkit detector.
    
    Checks for:
    - nvcc compiler version and location
    - libcudart runtime library version
    - CUDA_HOME environment variable
    - PATH configuration
    - LD_LIBRARY_PATH configuration (Linux)
    - Multiple CUDA installations
    - Driver compatibility
    """
    
    # Standard CUDA installation paths by platform
    CUDA_PATHS = {
        "Linux": [
            "/usr/local/cuda",
            "/usr/local/cuda-*",
            "/opt/cuda",
            "/usr/lib/cuda",
        ],
        "Windows": [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*",
            "C:\\CUDA\\*",
        ],
        "Darwin": [  # macOS (legacy, CUDA deprecated on macOS)
            "/usr/local/cuda",
            "/Developer/NVIDIA/CUDA-*",
        ]
    }
    
    def detect(self) -> DetectionResult:
        """
        Main detection logic that orchestrates all CUDA checks.
        
        Returns:
            DetectionResult with comprehensive CUDA toolkit information
        """
        result = DetectionResult(
            component="cuda_toolkit",
            status=Status.SUCCESS,
        )
        
        # 1. Find all CUDA installations
        installations = self._find_all_cuda_installations()
        
        if not installations:
            result.status = Status.NOT_FOUND
            result.add_recommendation(
                "Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
            )
            result.add_recommendation(
                "Or use pip-installed libraries with bundled CUDA (inference only)"
            )
            return result
        
        result.metadata["installations"] = installations
        result.metadata["installation_count"] = len(installations)
        
        # 2. Check nvcc (primary indicator)
        nvcc_info = self._check_nvcc()
        if nvcc_info:
            result.version = nvcc_info["version"]
            result.path = nvcc_info["path"]
            result.metadata["nvcc"] = nvcc_info
        else:
            result.add_issue("nvcc compiler not found in PATH")
            result.add_recommendation("Add CUDA bin directory to PATH")
            result.status = Status.WARNING
        
        # 3. Check CUDA_HOME
        cuda_home_info = self._check_cuda_home(installations)
        result.metadata["cuda_home"] = cuda_home_info
        
        if cuda_home_info["status"] == "missing":
            result.add_issue("CUDA_HOME environment variable not set")
            result.add_recommendation(
                f"Set CUDA_HOME to: {installations[0]['path']}"
            )
            if result.status == Status.SUCCESS:
                result.status = Status.WARNING
        elif cuda_home_info["status"] == "invalid":
            result.add_issue(f"CUDA_HOME points to non-existent path: {cuda_home_info['value']}")
            result.add_recommendation(
                f"Update CUDA_HOME to: {installations[0]['path']}"
            )
            result.status = Status.WARNING
        
        # 4. Check libcudart
        libcudart_info = self._check_libcudart(installations)
        result.metadata["libcudart"] = libcudart_info
        
        if not libcudart_info["found"]:
            result.add_issue("libcudart runtime library not found")
            result.status = Status.WARNING
        
        # 5. Check PATH configuration
        path_info = self._check_path_config(installations)
        result.metadata["path_config"] = path_info
        
        if not path_info["correct"]:
            result.add_issue("CUDA bin directory not in PATH or incorrect")
            result.add_recommendation(
                f"Add to PATH: {installations[0]['path']}/bin"
            )
            if result.status == Status.SUCCESS:
                result.status = Status.WARNING
        
        # 6. Check LD_LIBRARY_PATH (Linux only)
        if platform.system() == "Linux":
            ld_info = self._check_ld_library_path(installations)
            result.metadata["ld_library_path"] = ld_info
            
            if not ld_info["correct"]:
                result.add_issue("LD_LIBRARY_PATH not configured for CUDA")
                result.add_recommendation(
                    f"Add to LD_LIBRARY_PATH: {installations[0]['path']}/lib64"
                )
                if result.status == Status.SUCCESS:
                    result.status = Status.WARNING
        
        # 7. Check for multiple installations (potential conflict)
        if len(installations) > 1:
            result.metadata["multiple_installations"] = True
            result.add_issue(
                f"Multiple CUDA installations detected ({len(installations)}): "
                f"{', '.join([i['version'] for i in installations])}"
            )
            result.add_recommendation(
                "Ensure PATH and CUDA_HOME point to the desired version"
            )
            if result.status == Status.SUCCESS:
                result.status = Status.WARNING
        
        # 8. Validate driver compatibility
        driver_compat = self._check_driver_compatibility(result.version)
        result.metadata["driver_compatibility"] = driver_compat
        
        if not driver_compat["compatible"]:
            result.add_issue(driver_compat["message"])
            result.add_recommendation(
                "Upgrade GPU driver or downgrade CUDA toolkit"
            )
            result.status = Status.ERROR
        
        return result
    
    def _find_all_cuda_installations(self) -> List[Dict[str, str]]:
        """
        Find all CUDA toolkit installations on the system.
        
        Returns:
            List of dicts with 'path' and 'version' keys
        """
        installations = []
        system = platform.system()
        search_paths = self.CUDA_PATHS.get(system, [])
        
        found_paths = set()
        for pattern in search_paths:
            if "*" in pattern:
                # Glob expansion for wildcards
                matches = glob.glob(pattern)
                found_paths.update(matches)
            else:
                if os.path.exists(pattern):
                    found_paths.add(pattern)
        
        # Check symlinks (e.g., /usr/local/cuda -> /usr/local/cuda-11.8)
        resolved_paths = set()
        for path in found_paths:
            try:
                resolved = os.path.realpath(path)
                resolved_paths.add(resolved)
            except:
                resolved_paths.add(path)
        
        # Extract version from each installation
        for path in sorted(resolved_paths):
            version = self._extract_version_from_path(path)
            if version or self._validate_cuda_directory(path):
                installations.append({
                    "path": path,
                    "version": version or "Unknown"
                })
        
        return installations
    
    def _extract_version_from_path(self, path: str) -> Optional[str]:
        """Extract CUDA version from installation path."""
        # Match patterns like "cuda-11.8" or "v11.8"
        match = re.search(r'cuda[-_]?v?(\d+\.\d+)', path, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Try reading version.txt if it exists
        version_file = os.path.join(path, "version.txt")
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    content = f.read()
                    match = re.search(r'CUDA Version\s+(\d+\.\d+)', content)
                    if match:
                        return match.group(1)
            except:
                pass
        
        return None
    
    def _validate_cuda_directory(self, path: str) -> bool:
        """Check if directory looks like a valid CUDA installation."""
        # Check for key CUDA subdirectories/files
        indicators = [
            os.path.join(path, "bin", "nvcc"),
            os.path.join(path, "bin", "nvcc.exe"),
            os.path.join(path, "include", "cuda.h"),
            os.path.join(path, "lib64"),
            os.path.join(path, "lib"),
        ]
        return any(os.path.exists(ind) for ind in indicators)
    
    def _check_nvcc(self) -> Optional[Dict[str, str]]:
        """
        Check for nvcc compiler in PATH.
        
        Returns:
            Dict with version and path, or None if not found
        """
        # Try finding nvcc in PATH
        nvcc_path = shutil.which("nvcc")
        
        # Fallback: check common locations
        if not nvcc_path:
            fallback_paths = [
                "/usr/local/cuda/bin/nvcc",
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\bin\\nvcc.exe",
            ]
            for fallback in fallback_paths:
                if os.path.exists(fallback):
                    nvcc_path = fallback
                    break
        
        if not nvcc_path:
            return None
        
        try:
            result = subprocess.check_output(
                [nvcc_path, "--version"],
                encoding="utf-8",
                stderr=subprocess.STDOUT,
                timeout=5
            )
            match = re.search(r"release (\d+\.\d+)", result)
            if match:
                return {
                    "version": match.group(1),
                    "path": nvcc_path,
                    "found": True
                }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def _check_cuda_home(self, installations: List[Dict]) -> Dict:
        """
        Check CUDA_HOME environment variable.
        
        Returns:
            Dict with status: 'set', 'missing', or 'invalid'
        """
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        
        if not cuda_home:
            return {
                "status": "missing",
                "value": None
            }
        
        if not os.path.exists(cuda_home):
            return {
                "status": "invalid",
                "value": cuda_home
            }
        
        # Check if it matches one of the detected installations
        cuda_home_real = os.path.realpath(cuda_home)
        matches_installation = any(
            os.path.realpath(inst["path"]) == cuda_home_real
            for inst in installations
        )
        
        return {
            "status": "set",
            "value": cuda_home,
            "matches_installation": matches_installation
        }
    
    def _check_libcudart(self, installations: List[Dict]) -> Dict:
        """
        Check for libcudart runtime library.
        
        Returns:
            Dict with found status and version
        """
        system = platform.system()
        
        # Library name patterns by platform
        if system == "Linux":
            patterns = ["libcudart.so*"]
        elif system == "Windows":
            patterns = ["cudart64_*.dll", "cudart*.dll"]
        elif system == "Darwin":
            patterns = ["libcudart.dylib*"]
        else:
            return {"found": False, "reason": "Unsupported platform"}
        
        # Search in all installations
        for installation in installations:
            lib_dirs = [
                os.path.join(installation["path"], "lib64"),
                os.path.join(installation["path"], "lib"),
                os.path.join(installation["path"], "bin"),  # Windows
            ]
            
            for lib_dir in lib_dirs:
                if not os.path.exists(lib_dir):
                    continue
                
                for pattern in patterns:
                    matches = glob.glob(os.path.join(lib_dir, pattern))
                    if matches:
                        # Try to extract version
                        version = self._extract_libcudart_version(matches[0])
                        return {
                            "found": True,
                            "path": matches[0],
                            "version": version
                        }
        
        return {"found": False}
    
    def _extract_libcudart_version(self, lib_path: str) -> Optional[str]:
        """Extract version from libcudart filename or binary."""
        # Try filename first (e.g., cudart64_110.dll -> 11.0)
        filename = os.path.basename(lib_path)
        match = re.search(r'cudart(?:64)?[_-]?(\d+)', filename)
        if match:
            version_str = match.group(1)
            if len(version_str) == 3:  # e.g., "110" -> "11.0"
                return f"{version_str[0:2]}.{version_str[2]}"
            elif len(version_str) == 2:  # e.g., "11" -> "11.0"
                return f"{version_str}.0"
        
        # Try using system tools (Linux only)
        if platform.system() == "Linux":
            try:
                # Use readelf to extract SONAME
                result = subprocess.check_output(
                    ["readelf", "-d", lib_path],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                    timeout=5
                )
                match = re.search(r'libcudart\.so\.(\d+\.\d+)', result)
                if match:
                    return match.group(1)
            except:
                pass
        
        return "Unknown"
    
    def _check_path_config(self, installations: List[Dict]) -> Dict:
        """
        Check if CUDA bin directory is in PATH.
        
        Returns:
            Dict with correctness status
        """
        path_env = os.environ.get("PATH", "")
        path_dirs = path_env.split(os.pathsep)
        
        # Check if any CUDA bin directory is in PATH
        for installation in installations:
            cuda_bin = os.path.join(installation["path"], "bin")
            
            # Simple path string comparison (case-insensitive on Windows)
            for p in path_dirs:
                if not p:
                    continue
                # Normalize paths for comparison
                try:
                    normalized_cuda = os.path.normpath(cuda_bin).lower()
                    normalized_p = os.path.normpath(p).lower()
                    if normalized_cuda == normalized_p:
                        return {
                            "correct": True,
                            "cuda_bin_in_path": cuda_bin
                        }
                except:
                    # If normpath fails, try direct string comparison
                    if cuda_bin.lower() == p.lower():
                        return {
                            "correct": True,
                            "cuda_bin_in_path": cuda_bin
                        }
        
        return {
            "correct": False,
            "reason": "No CUDA bin directory found in PATH"
        }
    
    def _check_ld_library_path(self, installations: List[Dict]) -> Dict:
        """
        Check LD_LIBRARY_PATH configuration (Linux only).
        
        Returns:
            Dict with correctness status
        """
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        
        if not ld_path:
            return {
                "correct": False,
                "reason": "LD_LIBRARY_PATH not set"
            }
        
        ld_dirs = ld_path.split(":")
        
        # Check if any CUDA lib directory is in LD_LIBRARY_PATH
        for installation in installations:
            for lib_subdir in ["lib64", "lib"]:
                cuda_lib = os.path.join(installation["path"], lib_subdir)
                # Normalize paths for comparison
                try:
                    normalized_cuda = os.path.normpath(cuda_lib)
                    for ld_dir in ld_dirs:
                        if os.path.normpath(ld_dir) == normalized_cuda:
                            return {
                                "correct": True,
                                "cuda_lib_in_path": cuda_lib
                            }
                except:
                    # Fallback to direct string comparison
                    if cuda_lib in ld_dirs:
                        return {
                            "correct": True,
                            "cuda_lib_in_path": cuda_lib
                        }
        
        return {
            "correct": False,
            "reason": "No CUDA lib directory found in LD_LIBRARY_PATH",
            "value": ld_path
        }
    
    def _check_driver_compatibility(self, cuda_version: Optional[str]) -> Dict:
        """
        Check if CUDA toolkit is compatible with GPU driver.
        
        Returns:
            Dict with compatibility status
        """
        if not cuda_version or cuda_version == "Unknown":
            return {
                "compatible": True,  # Can't verify, assume OK
                "message": "CUDA version unknown, cannot verify compatibility"
            }
        
        # Try to get driver info from registry
        try:
            from env_doctor.core.registry import DetectorRegistry
            driver_detector = DetectorRegistry.get("nvidia_driver")
            driver_result = driver_detector.detect()
            
            if not driver_result.detected:
                return {
                    "compatible": True,  # No driver, can't verify
                    "message": "No NVIDIA driver detected, cannot verify compatibility"
                }
            
            max_cuda = driver_result.metadata.get("max_cuda_version", "Unknown")
            
            if max_cuda == "Unknown":
                return {
                    "compatible": True,
                    "message": "Driver max CUDA version unknown"
                }
            
            # Compare versions
            try:
                cuda_float = float(cuda_version)
                max_cuda_float = float(max_cuda)
                
                if cuda_float > max_cuda_float:
                    return {
                        "compatible": False,
                        "message": f"CUDA {cuda_version} requires driver supporting CUDA {cuda_version}, but driver only supports up to {max_cuda}",
                        "driver_version": driver_result.version,
                        "max_cuda": max_cuda,
                        "toolkit_cuda": cuda_version
                    }
                
                return {
                    "compatible": True,
                    "message": f"CUDA {cuda_version} is compatible with driver (supports up to {max_cuda})",
                    "driver_version": driver_result.version,
                    "max_cuda": max_cuda,
                    "toolkit_cuda": cuda_version
                }
            except ValueError:
                return {
                    "compatible": True,
                    "message": "Could not parse version numbers for comparison"
                }
        
        except Exception as e:
            return {
                "compatible": True,  # Can't verify, assume OK
                "message": f"Could not verify compatibility: {str(e)}"
            }