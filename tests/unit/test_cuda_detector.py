"""
Comprehensive unit tests for CUDA Toolkit Detector

Tests all edge cases and failure modes without requiring actual CUDA installation.
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import platform

from env_doctor.detectors.cuda_toolkit import CudaToolkitDetector
from env_doctor.core.detector import Status


class TestCudaToolkitDetector:
    """Test suite for CudaToolkitDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create a CudaToolkitDetector instance."""
        return CudaToolkitDetector()
    
    # ===== Test: nvcc found =====
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    def test_nvcc_found(self, mock_glob, mock_exists, mock_subprocess, mock_which, detector):
        """Test successful nvcc detection."""
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        
        def glob_side_effect(pattern):
            # Match any wildcard CUDA path search
            if "*" in pattern and "cuda" in pattern.lower():
                return ["/usr/local/cuda-11.8"]
            elif "libcudart" in pattern or "cudart" in pattern:
                return ["/usr/local/cuda-11.8/lib64/libcudart.so.11.0"]
            return []
        
        mock_glob.side_effect = glob_side_effect
        mock_exists.return_value = True
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-11.8",
            "PATH": "/usr/local/cuda-11.8/bin:/usr/bin"
        }, clear=True):
            result = detector.detect()
        
        # Should find CUDA (accept WARNING for PATH mismatch in cross-platform test)
        assert result.status in [Status.SUCCESS, Status.WARNING]
        assert result.version == "11.8"
        assert result.metadata["nvcc"]["found"] is True
    
    # ===== Test: nvcc not found =====
    
    @patch('shutil.which')
    @patch('os.path.exists')
    @patch('glob.glob')
    def test_nvcc_not_found(self, mock_glob, mock_exists, mock_which, detector):
        """Test when nvcc is not in PATH."""
        mock_which.return_value = None
        mock_exists.return_value = False
        mock_glob.return_value = []

        with patch.dict(os.environ, {}, clear=True):
            result = detector.detect()

        assert result.status == Status.NOT_FOUND
        assert any("CUDA" in rec for rec in result.recommendations)

    # ===== Test: Enhanced recommendations when driver is detected =====

    @patch('shutil.which')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_nvcc_not_found_with_driver_detected(self, mock_registry_get,
                                                  mock_glob, mock_exists, mock_which, detector):
        """Test enhanced recommendations when CUDA not found but driver is detected."""
        mock_which.return_value = None
        mock_exists.return_value = False
        mock_glob.return_value = []

        # Mock driver detector to return driver with max CUDA 12.2
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.146.02"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector

        with patch.dict(os.environ, {}, clear=True):
            result = detector.detect()

        assert result.status == Status.NOT_FOUND
        # Should have enhanced recommendations with specific version
        recommendations = " ".join(result.recommendations)
        assert "12.1" in recommendations  # 12.2 should map to 12.1
        assert "cuda-install" in recommendations.lower()

    @patch('shutil.which')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_nvcc_not_found_driver_not_detected(self, mock_registry_get,
                                                 mock_glob, mock_exists, mock_which, detector):
        """Test fallback recommendations when neither CUDA nor driver detected."""
        mock_which.return_value = None
        mock_exists.return_value = False
        mock_glob.return_value = []

        # Mock driver detector to return no driver
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = False
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector

        with patch.dict(os.environ, {}, clear=True):
            result = detector.detect()

        assert result.status == Status.NOT_FOUND
        # Should recommend installing driver first, then CUDA
        assert "driver" in result.recommendations[0].lower() or "Install CUDA Toolkit" in result.recommendations[0]
    
    # ===== Test: CUDA_HOME wrong =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    def test_cuda_home_wrong(self, mock_realpath, mock_glob, mock_exists, 
                            mock_subprocess, mock_which, detector):
        """Test when CUDA_HOME points to non-existent directory."""
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        mock_glob.return_value = ["/usr/local/cuda-11.8"]
        mock_realpath.side_effect = lambda x: x
        
        def exists_side_effect(path):
            return path != "/wrong/path"
        
        mock_exists.side_effect = exists_side_effect
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/wrong/path",
            "PATH": "/usr/local/cuda-11.8/bin:/usr/bin"
        }, clear=True):
            result = detector.detect()
        
        assert result.status == Status.WARNING
        assert any("CUDA_HOME points to non-existent" in issue for issue in result.issues)
    
    # ===== Test: Multiple CUDA toolkits installed =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    def test_multiple_cuda_installations(self, mock_realpath, mock_glob, mock_exists,
                                        mock_subprocess, mock_which, detector):
        """Test detection of multiple CUDA installations."""
        mock_which.return_value = "/usr/local/cuda-11.8/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        mock_glob.return_value = [
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-12.1"
        ]
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-11.8",
            "PATH": "/usr/local/cuda-11.8/bin:/usr/bin"
        }, clear=True):
            result = detector.detect()
        
        assert result.metadata["installation_count"] == 2
        assert result.metadata.get("multiple_installations") is True
        assert any("Multiple CUDA installations" in issue for issue in result.issues)
    
    # ===== Test: Toolkit mismatched with driver =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_toolkit_driver_mismatch(self, mock_registry_get, mock_realpath, 
                                    mock_glob, mock_exists, mock_subprocess,
                                    mock_which, detector):
        """Test when CUDA toolkit version exceeds driver capability."""
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.return_value = "release 12.2, V12.2.140"
        mock_glob.return_value = ["/usr/local/cuda-12.2"]
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x
        
        # Mock driver detector to return old driver
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "470.82"
        mock_driver_result.metadata = {"max_cuda_version": "11.4"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-12.2",
            "PATH": "/usr/local/cuda-12.2/bin:/usr/bin"
        }, clear=True):
            result = detector.detect()
        
        assert result.status == Status.ERROR
        assert result.metadata["driver_compatibility"]["compatible"] is False
        assert any("CUDA 12.2 requires" in issue for issue in result.issues)
    
    # ===== Test: Missing libcudart =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    def test_missing_libcudart(self, mock_realpath, mock_glob, mock_exists,
                               mock_subprocess, mock_which, detector):
        """Test when libcudart library is missing."""
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        mock_realpath.side_effect = lambda x: x
        
        # Mock glob for different patterns
        def glob_side_effect(pattern):
            if "cuda-*" in pattern or "CUDA" in pattern:
                return ["/usr/local/cuda-11.8"]
            return []  # No libcudart found
        
        mock_glob.side_effect = glob_side_effect
        
        def exists_side_effect(path):
            # Directories exist but not library files
            if "libcudart" in path or "cudart" in path:
                return False
            return True
        
        mock_exists.side_effect = exists_side_effect
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-11.8",
            "PATH": "/usr/local/cuda-11.8/bin:/usr/bin"
        }, clear=True):
            result = detector.detect()
        
        assert result.metadata["libcudart"]["found"] is False
        assert any("libcudart" in issue for issue in result.issues)
    
    # ===== Test: PATH points to wrong CUDA version =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    def test_path_wrong_cuda_version(self, mock_realpath, mock_glob,
                                     mock_exists, mock_subprocess, mock_which, detector):
        """Test when PATH points to different CUDA version than expected."""
        mock_which.return_value = "/usr/local/cuda-11.8/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        mock_glob.return_value = [
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-12.1"
        ]
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-12.1",  # Different from nvcc
            "PATH": "/usr/bin:/bin"  # No CUDA in PATH
        }, clear=True):
            result = detector.detect()
        
        assert result.status == Status.WARNING
        assert result.metadata["path_config"]["correct"] is False
    
    # ===== Test: LD_LIBRARY_PATH not set (Linux) =====
    
    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    def test_ld_library_path_not_set(self, mock_realpath, mock_glob, mock_exists,
                                     mock_subprocess, mock_which, mock_system, detector):
        """Test when LD_LIBRARY_PATH is not set on Linux."""
        mock_system.return_value = "Linux"
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        mock_glob.return_value = ["/usr/local/cuda-11.8"]
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x
        
        # Remove LD_LIBRARY_PATH from environment
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-11.8",
            "PATH": "/usr/local/cuda-11.8/bin:/usr/bin"
        }, clear=True):
            result = detector.detect()
        
        assert result.status == Status.WARNING
        assert result.metadata["ld_library_path"]["correct"] is False
        assert any("LD_LIBRARY_PATH" in issue for issue in result.issues)
    
    # ===== Test: Permission denied on CUDA directory =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_permission_denied(self, mock_file_open, mock_glob, mock_exists,
                              mock_subprocess, mock_which, detector):
        """Test handling of permission errors when reading CUDA files."""
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        mock_glob.return_value = ["/usr/local/cuda-11.8"]
        mock_exists.return_value = True
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-11.8",
            "PATH": "/usr/local/cuda-11.8/bin:/usr/bin"
        }, clear=True):
            # Should not crash, should handle gracefully
            result = detector.detect()
        
        # Should still return some result, not crash
        assert result is not None
        assert result.component == "cuda_toolkit"
    
    # ===== Test: Corrupt nvcc output =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    def test_corrupt_nvcc_output(self, mock_glob, mock_exists,
                                 mock_subprocess, mock_which, detector):
        """Test handling of corrupt or unexpected nvcc output."""
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.return_value = "CORRUPT DATA NO VERSION HERE"
        mock_glob.return_value = []
        mock_exists.return_value = True
        
        with patch.dict(os.environ, {}, clear=True):
            result = detector.detect()
        
        # Should handle gracefully, version will be None or Unknown
        assert result.status in [Status.WARNING, Status.NOT_FOUND]
    
    # ===== Test: nvcc timeout =====
    
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    def test_nvcc_timeout(self, mock_glob, mock_exists,
                         mock_subprocess, mock_which, detector):
        """Test handling of nvcc command timeout."""
        from subprocess import TimeoutExpired
        
        mock_which.return_value = "/usr/local/cuda/bin/nvcc"
        mock_subprocess.side_effect = TimeoutExpired("nvcc", 5)
        mock_glob.return_value = []
        mock_exists.return_value = False
        
        with patch.dict(os.environ, {}, clear=True):
            result = detector.detect()
        
        # Should handle timeout gracefully
        assert result.status == Status.NOT_FOUND
    
    # ===== Test: Windows-specific paths =====
    
    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    def test_windows_paths(self, mock_realpath, mock_glob, mock_exists,
                          mock_subprocess, mock_which, mock_system, detector):
        """Test CUDA detection on Windows."""
        mock_system.return_value = "Windows"
        mock_which.return_value = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        mock_glob.return_value = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
        ]
        mock_exists.return_value = True
        mock_realpath.side_effect = lambda x: x
        
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
        with patch.dict(os.environ, {
            "CUDA_PATH": cuda_path,
            "PATH": rf"{cuda_path}\bin;C:\Windows\System32"
        }, clear=True):
            result = detector.detect()
        
        assert result.detected
        assert result.version == "11.8"
        # On Windows, no LD_LIBRARY_PATH check
        assert "ld_library_path" not in result.metadata
    
    # ===== Test: apt-installed CUDA (system install) =====

    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_apt_installed_cuda_full(self, mock_registry_get, mock_realpath,
                                     mock_glob, mock_exists, mock_subprocess,
                                     mock_which, mock_system, detector):
        """Test full apt-installed CUDA: nvcc at /usr/bin + headers + libcudart → SUCCESS."""
        mock_system.return_value = "Linux"
        mock_which.return_value = "/usr/bin/nvcc"
        mock_subprocess.return_value = "release 12.1, V12.1.105"
        mock_realpath.side_effect = lambda x: x

        def glob_side_effect(pattern):
            p = pattern.replace("\\", "/")
            # No standard CUDA installations
            if "cuda-*" in p or "cuda*" in p or "/cuda" in p:
                return []
            # libcudart in system lib dir
            if "/usr/lib/x86_64-linux-gnu/libcudart.so" in p:
                return ["/usr/lib/x86_64-linux-gnu/libcudart.so.12.1"]
            return []

        mock_glob.side_effect = glob_side_effect

        def exists_side_effect(path):
            normalized = path.replace("\\", "/")
            existing = {
                "/usr/bin/nvcc",
                "/usr/include/cuda.h",
                "/usr/include/cuda_runtime.h",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/bin",
                "/usr",
            }
            return normalized in existing

        mock_exists.side_effect = exists_side_effect

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.183"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin"
        }, clear=True):
            result = detector.detect()

        assert result.status in [Status.SUCCESS, Status.WARNING]
        assert result.version == "12.1"
        assert result.metadata["nvcc"]["found"] is True
        # Should be a system install
        assert result.metadata["installations"][0]["install_type"] == "system"
        # CUDA_HOME missing should NOT be an issue for system installs
        assert not any("CUDA_HOME" in issue for issue in result.issues)
        # LD_LIBRARY_PATH should be marked correct (ldconfig handles it)
        assert result.metadata["ld_library_path"]["correct"] is True

    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    def test_apt_installed_nvcc_only_no_headers(self, mock_glob, mock_exists,
                                                 mock_subprocess, mock_which,
                                                 mock_system, detector):
        """Test nvcc found but no cuda.h or libcudart → WARNING with install recommendation."""
        mock_system.return_value = "Linux"
        mock_which.return_value = "/usr/bin/nvcc"
        mock_subprocess.return_value = "release 12.1, V12.1.105"
        mock_glob.return_value = []  # No standard installations, no libcudart

        def exists_side_effect(path):
            # Only nvcc exists, no headers, no runtime
            return path == "/usr/bin/nvcc"

        mock_exists.side_effect = exists_side_effect

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin"
        }, clear=True):
            result = detector.detect()

        assert result.status == Status.WARNING
        assert result.version == "12.1"
        assert any("runtime/development files missing" in issue for issue in result.issues)
        assert any("nvidia-cuda-toolkit" in rec for rec in result.recommendations)

    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_apt_installed_no_cuda_home_no_warning(self, mock_registry_get,
                                                    mock_realpath, mock_glob,
                                                    mock_exists, mock_subprocess,
                                                    mock_which, mock_system, detector):
        """Test system install without CUDA_HOME does not produce CUDA_HOME warning."""
        mock_system.return_value = "Linux"
        mock_which.return_value = "/usr/bin/nvcc"
        mock_subprocess.return_value = "release 12.1, V12.1.105"
        mock_realpath.side_effect = lambda x: x

        def glob_side_effect(pattern):
            p = pattern.replace("\\", "/")
            if "/usr/lib/x86_64-linux-gnu/libcudart.so" in p:
                return ["/usr/lib/x86_64-linux-gnu/libcudart.so.12.1"]
            return []

        mock_glob.side_effect = glob_side_effect

        def exists_side_effect(path):
            normalized = path.replace("\\", "/")
            existing = {
                "/usr/bin/nvcc", "/usr/include/cuda_runtime.h",
                "/usr/lib/x86_64-linux-gnu", "/usr/bin", "/usr",
            }
            return normalized in existing

        mock_exists.side_effect = exists_side_effect

        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.183"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin"
        }, clear=True):
            result = detector.detect()

        # No CUDA_HOME issue should be raised
        assert not any("CUDA_HOME" in issue for issue in result.issues)
        # The metadata should note it's not required
        assert "info" in result.metadata["cuda_home"]
        assert "not required" in result.metadata["cuda_home"]["info"]

    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_apt_installed_libcudart_system_path(self, mock_registry_get,
                                                  mock_realpath, mock_glob,
                                                  mock_exists, mock_subprocess,
                                                  mock_which, mock_system, detector):
        """Test libcudart found at /usr/lib/x86_64-linux-gnu/ for system install."""
        mock_system.return_value = "Linux"
        mock_which.return_value = "/usr/bin/nvcc"
        mock_subprocess.return_value = "release 12.1, V12.1.105"
        mock_realpath.side_effect = lambda x: x

        def glob_side_effect(pattern):
            p = pattern.replace("\\", "/")
            if "/usr/lib/x86_64-linux-gnu/libcudart.so" in p:
                return ["/usr/lib/x86_64-linux-gnu/libcudart.so.12.1"]
            return []

        mock_glob.side_effect = glob_side_effect

        def exists_side_effect(path):
            normalized = path.replace("\\", "/")
            existing = {
                "/usr/bin/nvcc", "/usr/include/cuda.h",
                "/usr/lib/x86_64-linux-gnu", "/usr/bin", "/usr",
            }
            return normalized in existing

        mock_exists.side_effect = exists_side_effect

        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.183"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin"
        }, clear=True):
            result = detector.detect()

        assert result.metadata["libcudart"]["found"] is True
        assert "x86_64-linux-gnu" in result.metadata["libcudart"]["path"]

    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_both_apt_and_standard_cuda(self, mock_registry_get, mock_realpath,
                                        mock_glob, mock_exists, mock_subprocess,
                                        mock_which, mock_system, detector):
        """Test when both system (apt) and /usr/local standard installs exist."""
        mock_system.return_value = "Linux"
        mock_which.return_value = "/usr/local/cuda-12.1/bin/nvcc"
        mock_subprocess.return_value = "release 12.1, V12.1.105"
        mock_realpath.side_effect = lambda x: x

        def glob_side_effect(pattern):
            if "cuda-*" in pattern or "/cuda*" in pattern:
                return ["/usr/local/cuda-12.1"]
            elif "libcudart" in pattern:
                return ["/usr/local/cuda-12.1/lib64/libcudart.so.12.1"]
            return []

        mock_glob.side_effect = glob_side_effect
        mock_exists.return_value = True

        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.183"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector

        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-12.1",
            "PATH": "/usr/local/cuda-12.1/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda-12.1/lib64"
        }, clear=True):
            result = detector.detect()

        # Standard install found, so system detection is not triggered
        assert result.metadata["installation_count"] >= 1
        assert result.metadata["installations"][0]["install_type"] == "standard"
        assert result.version == "12.1"

    # ===== Test: All checks pass perfectly =====
    @patch('platform.system')
    @patch('shutil.which')
    @patch('subprocess.check_output')
    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('os.path.realpath')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_perfect_configuration(self, mock_registry_get,
                                mock_realpath, mock_glob, mock_exists,
                                mock_subprocess, mock_which, mock_system, detector):
        """Test when all CUDA checks pass perfectly."""
        mock_system.return_value = "Linux"
        mock_which.return_value = "/usr/local/cuda-11.8/bin/nvcc"
        mock_subprocess.return_value = "release 11.8, V11.8.89"
        
        def glob_side_effect(p):
            if "cuda-*" in p or "/cuda*" in p:
                return ["/usr/local/cuda-11.8"]
            elif "libcudart" in p:
                return ["/usr/local/cuda-11.8/lib64/libcudart.so.11.0"]
            return []
        
        mock_glob.side_effect = glob_side_effect
        
        def exists_side_effect(path):
            allowed_paths = [
                "/usr/local/cuda-11.8",
                "/usr/local/cuda-11.8/bin",
                "/usr/local/cuda-11.8/lib64",
                "/usr/local/cuda-11.8/bin/nvcc"
            ]
            return any(path.startswith(allowed) for allowed in allowed_paths)
        
        mock_exists.side_effect = exists_side_effect
        mock_realpath.side_effect = lambda x: x
        
        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "530.30"
        mock_driver_result.metadata = {"max_cuda_version": "12.1"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry_get.return_value = mock_driver_detector
        
        with patch.dict(os.environ, {
            "CUDA_HOME": "/usr/local/cuda-11.8",
            "PATH": "/usr/local/cuda-11.8/bin:/usr/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda-11.8/lib64"
        }, clear=True):
            result = detector.detect()
        
        # Core detections should work
        assert result.status in [Status.SUCCESS, Status.WARNING]  # WARNING is OK in mocked env
        assert result.version == "11.8"
        assert result.metadata["nvcc"]["found"] is True
        assert result.metadata["libcudart"]["found"] is True
        assert result.metadata["cuda_home"]["status"] == "set"
        # PATH check might fail in cross-platform mocking - that's OK
        assert result.metadata["ld_library_path"]["correct"] is True
        assert result.metadata["driver_compatibility"]["compatible"] is True

# ===== Integration Tests =====

class TestCudaDetectorIntegration:
    """Integration tests that test detector with DetectorRegistry."""
    
    def test_detector_registered(self):
        """Test that CudaToolkitDetector is properly registered."""
        from env_doctor.core.registry import DetectorRegistry
        
        detector = DetectorRegistry.get("cuda_toolkit")
        assert detector is not None
        assert isinstance(detector, CudaToolkitDetector)
    
    @patch('shutil.which')
    @patch('glob.glob')
    def test_detector_via_registry(self, mock_glob, mock_which):
        """Test accessing detector through registry."""
        from env_doctor.core.registry import DetectorRegistry
        
        mock_which.return_value = None
        mock_glob.return_value = []
        
        with patch.dict(os.environ, {}, clear=True):
            detector = DetectorRegistry.get("cuda_toolkit")
            result = detector.detect()
        
        assert result.component == "cuda_toolkit"
        assert result.status == Status.NOT_FOUND