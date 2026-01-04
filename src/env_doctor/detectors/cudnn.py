"""
cuDNN Library Detector

Detects cuDNN (CUDA Deep Neural Network library) installations with comprehensive checks:
- Library file detection across standard paths (Linux and Windows)
- Version extraction from library
- Symlink validation (Linux) / DLL validation (Windows)
- Multiple installation detection
"""
import os
import re
import glob
import platform
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple

from env_doctor.core.detector import Detector, DetectionResult, Status
from env_doctor.core.registry import DetectorRegistry


@DetectorRegistry.register("cudnn")
class CudnnDetector(Detector):
    """
    Detects cuDNN library installation and validates configuration.

    cuDNN is NVIDIA's GPU-accelerated library for deep neural networks.
    This detector:
    - Searches standard installation paths (Linux and Windows)
    - Extracts version from library file
    - Validates symlinks (Linux) or DLL presence (Windows)
    - Provides actionable recommendations for issues

    Supported platforms: Linux, Windows
    """

    # Standard cuDNN library search paths - Linux
    CUDNN_SEARCH_PATHS_LINUX = [
        # Standard CUDA installation
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        # System-wide installation (apt/dpkg)
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        # WSL2 specific path
        "/usr/lib/wsl/lib",
        # Common alternative locations
        "/opt/cuda/lib64",
        "/usr/local/lib",
    ]

    # Standard cuDNN library search paths - Windows
    CUDNN_SEARCH_PATHS_WINDOWS = [
        # CUDA Toolkit default location
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\lib\x64",
        # cuDNN standalone install
        r"C:\Program Files\NVIDIA\CUDNN\*\bin",
        r"C:\tools\cuda\bin",
        # Common custom locations
        r"C:\CUDA\*\bin",
    ]

    # Library file patterns by platform
    CUDNN_PATTERNS_LINUX = [
        "libcudnn.so*",
        "libcudnn_*.so*",  # cuDNN 8.x has split libraries
    ]

    CUDNN_PATTERNS_WINDOWS = [
        "cudnn*.dll",
        "cudnn64_*.dll",
    ]

    def can_run(self) -> bool:
        """
        Check if this detector can run on the current platform.

        Returns:
            bool: True if running on Linux or Windows, False otherwise.
        """
        return platform.system() in ("Linux", "Windows")

    def detect(self) -> DetectionResult:
        """
        Perform cuDNN detection and validation.

        Searches for cuDNN libraries, extracts version information,
        and validates symlink configuration (Linux) or DLL presence (Windows).

        Returns:
            DetectionResult: Detection outcome with version, issues, and recommendations.
        """
        result = DetectionResult(
            component="cudnn",
            status=Status.SUCCESS,
        )

        system = platform.system()
        result.metadata["platform"] = system

        # 1. Find all cuDNN libraries
        libraries = self._find_cudnn_libraries()

        if not libraries:
            result.status = Status.NOT_FOUND
            result.add_issue("cuDNN library not found on system")
            result.add_recommendation(
                "Install cuDNN from: https://developer.nvidia.com/cudnn"
            )
            if system == "Linux":
                result.add_recommendation(
                    "For Ubuntu/Debian: sudo apt install libcudnn8 libcudnn8-dev"
                )
            else:
                result.add_recommendation(
                    "Extract cuDNN to CUDA toolkit directory or add to PATH"
                )
            return result

        result.metadata["libraries"] = libraries
        result.metadata["library_count"] = len(libraries)

        # 2. Get the primary library
        primary_lib = self._select_primary_library(libraries)
        result.path = primary_lib["path"]
        result.metadata["primary_library"] = primary_lib

        # 3. Extract version
        version = self._extract_version(primary_lib["path"])
        if version:
            result.version = version
            result.metadata["version_source"] = primary_lib.get("version_source", "detected")
        else:
            result.version = "Unknown"
            result.add_issue("Could not extract cuDNN version from library")
            if result.status == Status.SUCCESS:
                result.status = Status.WARNING

        # 4. Check symlinks (Linux) or PATH configuration (Windows)
        if system == "Linux":
            symlink_issues = self._check_symlinks(libraries)
            result.metadata["symlink_status"] = symlink_issues

            if symlink_issues["missing"]:
                result.add_issue(
                    f"Missing cuDNN symlinks: {', '.join(symlink_issues['missing'])}"
                )
                result.add_recommendation(
                    "Create missing symlinks or reinstall cuDNN package"
                )
                if result.status == Status.SUCCESS:
                    result.status = Status.WARNING

            if symlink_issues["broken"]:
                result.add_issue(
                    f"Broken cuDNN symlinks: {', '.join(symlink_issues['broken'])}"
                )
                result.add_recommendation(
                    "Fix broken symlinks: sudo ldconfig or reinstall cuDNN"
                )
                result.status = Status.ERROR
        else:
            # Windows: check if DLL is in PATH
            path_status = self._check_windows_path(libraries)
            result.metadata["path_status"] = path_status

            if not path_status["in_path"]:
                result.add_issue("cuDNN DLL directory not in system PATH")
                result.add_recommendation(
                    f"Add to PATH: {path_status.get('suggested_path', 'cuDNN bin directory')}"
                )
                if result.status == Status.SUCCESS:
                    result.status = Status.WARNING

        # 5. Check for multiple versions (potential conflict)
        versions_found = self._detect_multiple_versions(libraries)
        if len(versions_found) > 1:
            result.metadata["multiple_versions"] = versions_found
            result.add_issue(
                f"Multiple cuDNN versions detected: {', '.join(versions_found)}"
            )
            result.add_recommendation(
                "Consider removing old cuDNN versions to avoid conflicts"
            )
            if result.status == Status.SUCCESS:
                result.status = Status.WARNING

        # 6. Check CUDA compatibility
        cuda_compat = self._check_cuda_compatibility(result.version)
        result.metadata["cuda_compatibility"] = cuda_compat

        if cuda_compat and not cuda_compat.get("compatible", True):
            result.add_issue(cuda_compat.get("message", "CUDA compatibility issue"))
            result.add_recommendation(
                "Ensure cuDNN version matches your CUDA toolkit version"
            )
            if result.status == Status.SUCCESS:
                result.status = Status.WARNING

        return result

    def _get_search_paths(self) -> List[str]:
        """
        Get platform-specific search paths.

        Returns:
            List of directory paths to search.
        """
        if platform.system() == "Windows":
            return self.CUDNN_SEARCH_PATHS_WINDOWS
        return self.CUDNN_SEARCH_PATHS_LINUX

    def _get_library_patterns(self) -> List[str]:
        """
        Get platform-specific library file patterns.

        Returns:
            List of glob patterns for library files.
        """
        if platform.system() == "Windows":
            return self.CUDNN_PATTERNS_WINDOWS
        return self.CUDNN_PATTERNS_LINUX

    def _find_cudnn_libraries(self) -> List[Dict]:
        """
        Search for cuDNN library files across standard paths.

        Returns:
            List of dicts with library information (path, filename, is_symlink).
        """
        libraries = []
        seen_paths = set()
        search_paths = self._get_search_paths()
        patterns = self._get_library_patterns()

        # Expand wildcards in search paths first (for Windows CUDA version dirs)
        expanded_paths = []
        for search_path in search_paths:
            if "*" in search_path:
                expanded = glob.glob(search_path)
                expanded_paths.extend(expanded)
            else:
                expanded_paths.append(search_path)

        for search_path in expanded_paths:
            if not os.path.exists(search_path):
                continue

            for pattern in patterns:
                full_pattern = os.path.join(search_path, pattern)
                matches = glob.glob(full_pattern)

                for match in matches:
                    if match in seen_paths:
                        continue

                    # Resolve real path (for symlinks on Linux)
                    try:
                        real_path = os.path.realpath(match)
                    except OSError:
                        real_path = match

                    lib_info = {
                        "path": match,
                        "real_path": real_path,
                        "filename": os.path.basename(match),
                        "is_symlink": os.path.islink(match),
                        "directory": os.path.dirname(match),
                    }

                    # Check if readable
                    lib_info["readable"] = os.access(match, os.R_OK)

                    libraries.append(lib_info)
                    seen_paths.add(match)

        return libraries

    def _select_primary_library(self, libraries: List[Dict]) -> Dict:
        """
        Select the primary cuDNN library from found libraries.

        Prefers versioned files over symlinks on Linux,
        prefers main cudnn64_*.dll on Windows.

        Args:
            libraries: List of library info dicts.

        Returns:
            Dict with primary library information.
        """
        system = platform.system()

        if system == "Windows":
            # Prefer cudnn64_*.dll (main library)
            main_dlls = [
                lib for lib in libraries
                if re.match(r'cudnn64_\d+\.dll', lib["filename"], re.IGNORECASE)
            ]
            if main_dlls:
                main_dlls.sort(
                    key=lambda x: self._extract_version_numbers(x["filename"]),
                    reverse=True
                )
                return main_dlls[0]
        else:
            # Linux: prefer non-symlink, versioned files
            versioned = [
                lib for lib in libraries
                if not lib["is_symlink"] and re.search(r'\.so\.\d+', lib["filename"])
            ]
            if versioned:
                versioned.sort(
                    key=lambda x: self._extract_version_numbers(x["filename"]),
                    reverse=True
                )
                return versioned[0]

        # Fall back to any readable library
        readable = [lib for lib in libraries if lib["readable"]]
        if readable:
            return readable[0]

        # Last resort: return first found
        return libraries[0]

    def _extract_version_numbers(self, filename: str) -> Tuple[int, ...]:
        """
        Extract version numbers from filename for sorting.

        Args:
            filename: Library filename (e.g., libcudnn.so.8.9.0 or cudnn64_8.dll).

        Returns:
            Tuple of version numbers for sorting.
        """
        # Linux pattern: libcudnn.so.8.9.0
        match = re.search(r'\.so\.(\d+)(?:\.(\d+))?(?:\.(\d+))?', filename)
        if match:
            parts = [int(p) if p else 0 for p in match.groups()]
            return tuple(parts)

        # Windows pattern: cudnn64_8.dll or cudnn_ops_infer64_8.dll
        match = re.search(r'_(\d+)\.dll', filename, re.IGNORECASE)
        if match:
            return (int(match.group(1)), 0, 0)

        return (0, 0, 0)

    def _extract_version(self, lib_path: str) -> Optional[str]:
        """
        Extract cuDNN version from library file.

        Uses platform-specific methods:
        - Linux: readelf, filename parsing, strings
        - Windows: dumpbin, filename parsing

        Args:
            lib_path: Path to cuDNN library file.

        Returns:
            Version string (e.g., "8.9.0" or "8") or None if extraction fails.
        """
        system = platform.system()

        if system == "Linux":
            # Try readelf first (most accurate)
            version = self._extract_version_readelf(lib_path)
            if version:
                return version

        # Try filename parsing
        version = self._extract_version_from_filename(lib_path)
        if version:
            return version

        if system == "Windows":
            # Try dumpbin for Windows
            version = self._extract_version_dumpbin(lib_path)
            if version:
                return version

        if system == "Linux":
            # Try strings command as last resort on Linux
            return self._extract_version_strings(lib_path)

        return None

    def _extract_version_readelf(self, lib_path: str) -> Optional[str]:
        """
        Extract version using readelf -d (dynamic section) - Linux only.

        Args:
            lib_path: Path to library file.

        Returns:
            Version string or None.
        """
        try:
            result = subprocess.check_output(
                ["readelf", "-d", lib_path],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
                timeout=5
            )

            # Look for SONAME entry: libcudnn.so.8 or libcudnn.so.8.9.0
            match = re.search(r'libcudnn[^.]*\.so\.(\d+(?:\.\d+)*)', result)
            if match:
                return match.group(1)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError):
            pass

        return None

    def _extract_version_dumpbin(self, lib_path: str) -> Optional[str]:
        """
        Extract version using dumpbin /headers - Windows only.

        Args:
            lib_path: Path to DLL file.

        Returns:
            Version string or None.
        """


        if not shutil.which("dumpbin"):
            return None  # dumpbin not available, skip

        try:
            # dumpbin is part of Visual Studio, might not be available
            result = subprocess.check_output(
                ["dumpbin", "/headers", lib_path],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
                timeout=10
            )

            # Look for file version in headers
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)\s+file version', result, re.IGNORECASE)
            if match:
                parts = match.group(1).split(".")
                # cuDNN version is typically major.minor.patch
                if len(parts) >= 3:
                    return f"{parts[0]}.{parts[1]}.{parts[2]}"

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError):
            pass

        return None

    def _extract_version_from_filename(self, lib_path: str) -> Optional[str]:
        """
        Extract version from library filename.

        Args:
            lib_path: Path to library file.

        Returns:
            Version string or None.
        """
        filename = os.path.basename(lib_path)

        # Linux: libcudnn.so.8.9.0 or libcudnn.so.8
        match = re.search(r'libcudnn[^.]*\.so\.(\d+(?:\.\d+)*)', filename)
        if match:
            return match.group(1)

        # Windows: cudnn64_8.dll -> version 8
        match = re.search(r'cudnn(?:64)?_(\d+)\.dll', filename, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _extract_version_strings(self, lib_path: str) -> Optional[str]:
        """
        Extract version using strings command - Linux only (last resort).

        Args:
            lib_path: Path to library file.

        Returns:
            Version string or None.
        """
        try:
            result = subprocess.check_output(
                ["strings", lib_path],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
                timeout=10
            )

            # Look for cuDNN version pattern
            match = re.search(r'cuDNN[^\d]*(\d+\.\d+\.\d+)', result, re.IGNORECASE)
            if match:
                return match.group(1)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError):
            pass

        return None

    def _check_symlinks(self, libraries: List[Dict]) -> Dict:
        """
        Check for missing or broken cuDNN symlinks - Linux only.

        TensorFlow and PyTorch often look for specific symlink names
        like libcudnn.so (without version).

        Args:
            libraries: List of library info dicts.

        Returns:
            Dict with 'missing' and 'broken' symlink lists.
        """
        result = {
            "missing": [],
            "broken": [],
            "valid": [],
        }

        # Find directories that have cuDNN libraries
        lib_dirs = set(lib["directory"] for lib in libraries)

        for lib_dir in lib_dirs:
            # Check for expected symlinks
            expected_symlinks = [
                "libcudnn.so",  # Generic symlink (most important)
            ]

            for symlink_name in expected_symlinks:
                symlink_path = os.path.join(lib_dir, symlink_name)

                if os.path.islink(symlink_path):
                    # Check if symlink target exists
                    if os.path.exists(symlink_path):
                        result["valid"].append(symlink_path)
                    else:
                        result["broken"].append(symlink_path)
                elif os.path.exists(symlink_path):
                    # Regular file (not a symlink) - this is fine
                    result["valid"].append(symlink_path)
                else:
                    # Check if there's a versioned library that should have a symlink
                    versioned_libs = glob.glob(os.path.join(lib_dir, "libcudnn.so.*"))
                    if versioned_libs:
                        result["missing"].append(symlink_path)

        return result

    def _check_windows_path(self, libraries: List[Dict]) -> Dict:
        """Check if cuDNN DLL directory is in system PATH - Windows only."""
        path_env = os.environ.get("PATH", "")
        path_dirs = path_env.split(os.pathsep)
        
        # Normalize all PATH directories
        normalized_paths = [os.path.normpath(p).lower() for p in path_dirs]
        
        # Get unique directories containing cuDNN DLLs
        cudnn_dirs = set(lib["directory"] for lib in libraries)
        
        for cudnn_dir in cudnn_dirs:
            normalized_cudnn = os.path.normpath(cudnn_dir).lower()
            if normalized_cudnn in normalized_paths:
                return {
                    "in_path": True,
                    "directory": cudnn_dir
                }
        
        # Not in PATH
        return {
            "in_path": False,
            "suggested_path": list(cudnn_dirs)[0] if cudnn_dirs else None
        }


    def _detect_multiple_versions(self, libraries: List[Dict]) -> List[str]:
        """
        Detect multiple cuDNN versions installed.

        Args:
            libraries: List of library info dicts.

        Returns:
            List of unique version strings found.
        """
        versions = set()

        for lib in libraries:
            # Skip symlinks to avoid counting same version multiple times
            if lib["is_symlink"]:
                continue

            version = self._extract_version_from_filename(lib["path"])
            if version:
                # Normalize to major version for comparison
                parts = version.split(".")
                if parts:
                    versions.add(parts[0])

        return sorted(versions, reverse=True)

    def _check_cuda_compatibility(self, cudnn_version: Optional[str]) -> Dict:
        """
        Check if cuDNN version is compatible with installed CUDA toolkit.

        Args:
            cudnn_version: Detected cuDNN version string.

        Returns:
            Dict with compatibility information.
        """
        if not cudnn_version or cudnn_version == "Unknown":
            return {"compatible": True, "message": "Cannot verify - unknown cuDNN version"}

        try:
            from env_doctor.core.registry import DetectorRegistry
            cuda_detector = DetectorRegistry.get("cuda_toolkit")
            cuda_result = cuda_detector.detect()

            if not cuda_result.detected:
                return {
                    "compatible": True,
                    "message": "Cannot verify - CUDA toolkit not detected"
                }

            cuda_version = cuda_result.version
            if not cuda_version or cuda_version == "Unknown":
                return {
                    "compatible": True,
                    "message": "Cannot verify - unknown CUDA version"
                }

            # cuDNN 8.x supports CUDA 10.2, 11.x, 12.x
            # cuDNN 9.x supports CUDA 12.x
            cudnn_major = int(cudnn_version.split(".")[0])
            cuda_major = int(cuda_version.split(".")[0])

            compatible = True
            message = f"cuDNN {cudnn_version} with CUDA {cuda_version}"

            if cudnn_major >= 9 and cuda_major < 12:
                compatible = False
                message = f"cuDNN {cudnn_version} requires CUDA 12.x (found {cuda_version})"
            elif cudnn_major == 8 and cuda_major < 10:
                compatible = False
                message = f"cuDNN {cudnn_version} requires CUDA 10.2+ (found {cuda_version})"

            return {
                "compatible": compatible,
                "message": message,
                "cudnn_version": cudnn_version,
                "cuda_version": cuda_version
            }

        except Exception as e:
            return {
                "compatible": True,
                "message": f"Cannot verify compatibility: {str(e)}"
            }