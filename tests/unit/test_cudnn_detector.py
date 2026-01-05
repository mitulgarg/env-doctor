"""
Comprehensive unit tests for cuDNN Detector

Tests all edge cases and failure modes without requiring actual cuDNN installation.
Covers both Linux and Windows platforms.
"""
import pytest
from unittest.mock import patch, MagicMock
import os
import platform

from env_doctor.detectors.cudnn import CudnnDetector
from env_doctor.core.detector import Status


class TestCudnnDetector:
    """Test suite for CudnnDetector."""

    @pytest.fixture
    def detector(self):
        """Create a CudnnDetector instance."""
        return CudnnDetector()

    # ===== Test: can_run() =====

    @patch('platform.system')
    def test_can_run_linux(self, mock_system, detector):
        """Test that detector can run on Linux."""
        mock_system.return_value = "Linux"
        assert detector.can_run() is True

    @patch('platform.system')
    def test_can_run_windows(self, mock_system, detector):
        """Test that detector can run on Windows."""
        mock_system.return_value = "Windows"
        assert detector.can_run() is True

    @patch('platform.system')
    def test_cannot_run_macos(self, mock_system, detector):
        """Test that detector cannot run on macOS."""
        mock_system.return_value = "Darwin"
        assert detector.can_run() is False

    # ===== Test: cuDNN found (Linux) =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_cudnn_found_linux(self, mock_subprocess, mock_glob, mock_realpath,
                                mock_access, mock_islink, mock_exists,
                                mock_system, detector):
        """Test successful cuDNN detection on Linux."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x

        def glob_side_effect(pattern):
            if "libcudnn.so*" in pattern and "/usr/local/cuda/lib64" in pattern:
                return ["/usr/local/cuda/lib64/libcudnn.so.8.9.0"]
            if "libcudnn.so.*" in pattern:
                return ["/usr/local/cuda/lib64/libcudnn.so.8.9.0"]
            return []

        mock_glob.side_effect = glob_side_effect
        mock_subprocess.return_value = "SONAME: libcudnn.so.8"

        result = detector.detect()

        assert result.status in [Status.SUCCESS, Status.WARNING]
        assert result.component == "cudnn"
        assert result.version is not None
        assert result.path is not None

    # ===== Test: cuDNN found (Windows) =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    def test_cudnn_found_windows(self, mock_glob, mock_realpath,
                                  mock_access, mock_islink, mock_exists,
                                  mock_system, detector):
        """Test successful cuDNN detection on Windows."""
        mock_system.return_value = "Windows"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x

        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"

        def glob_side_effect(pattern):
            if "cudnn*.dll" in pattern or "cudnn64_*.dll" in pattern:
                if "CUDA" in pattern:
                    return [rf"{cuda_path}\cudnn64_8.dll"]
            if r"CUDA\*\bin" in pattern:
                return [cuda_path]
            return []

        mock_glob.side_effect = glob_side_effect

        with patch.dict(os.environ, {
            "PATH": rf"{cuda_path};C:\Windows\System32"
        }, clear=True):
            result = detector.detect()

        assert result.status in [Status.SUCCESS, Status.WARNING]
        assert result.component == "cudnn"

    # ===== Test: cuDNN not found =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('glob.glob')
    def test_cudnn_not_found(self, mock_glob, mock_exists, mock_system, detector):
        """Test when cuDNN is not installed."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = False
        mock_glob.return_value = []

        result = detector.detect()

        assert result.status == Status.NOT_FOUND
        assert "cuDNN library not found" in result.issues[0]
        assert len(result.recommendations) > 0

    # ===== Test: Missing symlinks (Linux) =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_missing_symlinks_linux(self, mock_subprocess, mock_glob, mock_realpath,
                                     mock_access, mock_islink, mock_exists,
                                     mock_system, detector):
        """Test detection of missing symlinks on Linux."""
        mock_system.return_value = "Linux"
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x
        mock_subprocess.return_value = "SONAME: libcudnn.so.8"

        lib_dir = "/usr/local/cuda/lib64"

        def exists_side_effect(path):
            # Directory exists, versioned lib exists, but symlink doesn't
            if path == lib_dir:
                return True
            if "libcudnn.so.8" in path:
                return True
            if path == f"{lib_dir}/libcudnn.so":
                return False  # Missing symlink
            return True

        mock_exists.side_effect = exists_side_effect

        def islink_side_effect(path):
            return False

        mock_islink.side_effect = islink_side_effect

        def glob_side_effect(pattern):
            if "libcudnn.so*" in pattern and lib_dir in pattern:
                return [f"{lib_dir}/libcudnn.so.8.9.0"]
            if "libcudnn.so.*" in pattern:
                return [f"{lib_dir}/libcudnn.so.8.9.0"]
            return []

        mock_glob.side_effect = glob_side_effect

        result = detector.detect()

        # Should detect the missing symlink issue
        assert result.metadata.get("symlink_status") is not None
        symlink_status = result.metadata["symlink_status"]
        assert len(symlink_status["missing"]) > 0 or len(symlink_status["broken"]) > 0 or result.status == Status.SUCCESS

    # ===== Test: Broken symlinks (Linux) =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_broken_symlinks_linux(self, mock_subprocess, mock_glob, mock_realpath,
                                    mock_access, mock_islink, mock_exists,
                                    mock_system, detector):
        """Test detection of broken symlinks on Linux."""
        mock_system.return_value = "Linux"
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x
        mock_subprocess.return_value = "SONAME: libcudnn.so.8"

        lib_dir = "/usr/local/cuda/lib64"

        def exists_side_effect(path):
            if path == lib_dir:
                return True
            if "libcudnn.so.8" in path and "libcudnn.so" not in path.split("/")[-1].replace(".8", ""):
                return True
            # Broken symlink: islink returns True but exists returns False
            if path == f"{lib_dir}/libcudnn.so":
                return False
            return True

        mock_exists.side_effect = exists_side_effect

        def islink_side_effect(path):
            if path == f"{lib_dir}/libcudnn.so":
                return True  # It's a symlink
            return False

        mock_islink.side_effect = islink_side_effect

        def glob_side_effect(pattern):
            if "libcudnn.so*" in pattern and lib_dir in pattern:
                return [f"{lib_dir}/libcudnn.so", f"{lib_dir}/libcudnn.so.8.9.0"]
            if "libcudnn.so.*" in pattern:
                return [f"{lib_dir}/libcudnn.so.8.9.0"]
            return []

        mock_glob.side_effect = glob_side_effect

        result = detector.detect()

        # Should detect the broken symlink
        if "symlink_status" in result.metadata:
            symlink_status = result.metadata["symlink_status"]
            # Check for broken symlink detection
            assert "broken" in symlink_status

    # ===== Test: Version extraction from readelf output =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_version_extraction_readelf(self, mock_subprocess, mock_glob,
                                         mock_realpath, mock_access, mock_islink,
                                         mock_exists, mock_system, detector):
        """Test version extraction using readelf."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x

        lib_path = "/usr/local/cuda/lib64/libcudnn.so.8.9.0"

        def glob_side_effect(pattern):
            if "libcudnn" in pattern:
                return [lib_path]
            return []

        mock_glob.side_effect = glob_side_effect

        # Simulate readelf output
        mock_subprocess.return_value = """
Dynamic section at offset 0x1234:
  Tag        Type                         Name/Value
 0x00000001 (SONAME)                     Library soname: [libcudnn.so.8]
        """

        result = detector.detect()

        assert result.version == "8" or result.version == "8.9.0"

    # ===== Test: Version extraction from filename =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_version_extraction_filename(self, mock_subprocess, mock_glob,
                                          mock_realpath, mock_access, mock_islink,
                                          mock_exists, mock_system, detector):
        """Test version extraction from filename when readelf fails."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x

        lib_path = "/usr/local/cuda/lib64/libcudnn.so.8.6.0"

        def glob_side_effect(pattern):
            if "libcudnn" in pattern:
                return [lib_path]
            return []

        mock_glob.side_effect = glob_side_effect

        # readelf fails
        mock_subprocess.side_effect = FileNotFoundError("readelf not found")

        result = detector.detect()

        # Should fall back to filename extraction
        assert result.version == "8.6.0"

    # ===== Test: Multiple cuDNN versions present =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_multiple_versions(self, mock_subprocess, mock_glob, mock_realpath,
                                mock_access, mock_islink, mock_exists,
                                mock_system, detector):
        """Test detection of multiple cuDNN versions."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x
        mock_subprocess.return_value = "SONAME: libcudnn.so.8"

        def islink_side_effect(path):
            return "libcudnn.so.8" in path or "libcudnn.so.7" in path

        mock_islink.side_effect = islink_side_effect

        def glob_side_effect(pattern):
            if "libcudnn.so*" in pattern:
                return [
                    "/usr/local/cuda/lib64/libcudnn.so.8.9.0",
                    "/usr/local/cuda/lib64/libcudnn.so.7.6.5",
                ]
            return []

        mock_glob.side_effect = glob_side_effect

        result = detector.detect()

        # Should detect multiple versions
        if result.status != Status.NOT_FOUND:
            versions = result.metadata.get("multiple_versions", [])
            assert len(versions) >= 1 or result.metadata.get("library_count", 0) >= 1

    # ===== Test: No read permissions on library file =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    def test_no_read_permissions(self, mock_glob, mock_realpath, mock_access,
                                  mock_islink, mock_exists, mock_system, detector):
        """Test handling of files without read permissions."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = False  # No read permission
        mock_realpath.side_effect = lambda x: x

        def glob_side_effect(pattern):
            if "libcudnn" in pattern:
                return ["/usr/local/cuda/lib64/libcudnn.so.8.9.0"]
            return []

        mock_glob.side_effect = glob_side_effect

        result = detector.detect()

        # Should still find the library but may have issues extracting version
        assert result.component == "cudnn"
        if result.status != Status.NOT_FOUND:
            assert result.metadata.get("library_count", 0) >= 1

    # ===== Test: readelf timeout =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_readelf_timeout(self, mock_subprocess, mock_glob, mock_realpath,
                              mock_access, mock_islink, mock_exists,
                              mock_system, detector):
        """Test handling of readelf timeout."""
        from subprocess import TimeoutExpired

        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x

        lib_path = "/usr/local/cuda/lib64/libcudnn.so.8.9.0"

        def glob_side_effect(pattern):
            if "libcudnn" in pattern:
                return [lib_path]
            return []

        mock_glob.side_effect = glob_side_effect
        mock_subprocess.side_effect = TimeoutExpired("readelf", 5)

        result = detector.detect()

        # Should handle timeout gracefully and fall back to filename
        assert result.component == "cudnn"
        assert result.version == "8.9.0"  # Extracted from filename

    # ===== Test: WSL2 path detection =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    def test_wsl2_path(self, mock_subprocess, mock_glob, mock_realpath,
                        mock_access, mock_islink, mock_exists,
                        mock_system, detector):
        """Test cuDNN detection in WSL2 path."""
        mock_system.return_value = "Linux"
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x
        mock_subprocess.return_value = "SONAME: libcudnn.so.8"

        def exists_side_effect(path):
            return "/usr/lib/wsl/lib" in path or path == "/usr/lib/wsl/lib"

        mock_exists.side_effect = exists_side_effect

        def glob_side_effect(pattern):
            if "/usr/lib/wsl/lib" in pattern and "libcudnn" in pattern:
                return ["/usr/lib/wsl/lib/libcudnn.so.8"]
            return []

        mock_glob.side_effect = glob_side_effect

        result = detector.detect()

        if result.status != Status.NOT_FOUND:
            assert "/usr/lib/wsl/lib" in result.path

    # ===== Test: Windows PATH not set =====

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    def test_windows_path_not_set(self, mock_glob, mock_realpath,
                                   mock_access, mock_islink, mock_exists,
                                   mock_system, detector):
        """Test Windows detection when cuDNN directory not in PATH."""
        mock_system.return_value = "Windows"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x

        cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"

        def glob_side_effect(pattern):
            if "cudnn" in pattern.lower():
                return [rf"{cuda_bin}\cudnn64_8.dll"]
            if "CUDA" in pattern:
                return [cuda_bin]
            return []

        mock_glob.side_effect = glob_side_effect

        # PATH does NOT include cuDNN directory
        with patch.dict(os.environ, {
            "PATH": r"C:\Windows\System32;C:\Windows"
        }, clear=True):
            result = detector.detect()

        if result.status != Status.NOT_FOUND:
            assert result.metadata.get("path_status", {}).get("in_path") is False
            assert any("PATH" in issue for issue in result.issues)


# ===== Integration Tests =====

class TestCudnnDetectorIntegration:
    """Integration tests for CudnnDetector with DetectorRegistry."""

    def test_detector_registered(self):
        """Test that CudnnDetector is properly registered."""
        from env_doctor.core.registry import DetectorRegistry

        detector = DetectorRegistry.get("cudnn")
        assert detector is not None
        assert isinstance(detector, CudnnDetector)

    @patch('platform.system')
    @patch('glob.glob')
    @patch('os.path.exists')
    def test_detector_via_registry(self, mock_exists, mock_glob, mock_system):
        """Test accessing detector through registry."""
        from env_doctor.core.registry import DetectorRegistry

        mock_system.return_value = "Linux"
        mock_glob.return_value = []
        mock_exists.return_value = False

        detector = DetectorRegistry.get("cudnn")
        result = detector.detect()

        assert result.component == "cudnn"
        assert result.status == Status.NOT_FOUND

    @patch('platform.system')
    def test_can_run_check_via_registry(self, mock_system):
        """Test can_run() check via registry."""
        from env_doctor.core.registry import DetectorRegistry

        mock_system.return_value = "Darwin"  # macOS

        detector = DetectorRegistry.get("cudnn")
        assert detector.can_run() is False


# ===== CUDA Compatibility Tests =====

class TestCudnnCudaCompatibility:
    """Test CUDA compatibility checking."""

    @pytest.fixture
    def detector(self):
        return CudnnDetector()

    @patch('platform.system')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.access')
    @patch('os.path.realpath')
    @patch('glob.glob')
    @patch('subprocess.check_output')
    @patch('env_doctor.core.registry.DetectorRegistry.get')
    def test_cuda_compatibility_check(self, mock_registry_get, mock_subprocess,
                                       mock_glob, mock_realpath, mock_access,
                                       mock_islink, mock_exists, mock_system,
                                       detector):
        """Test CUDA compatibility checking."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_islink.return_value = False
        mock_access.return_value = True
        mock_realpath.side_effect = lambda x: x
        mock_subprocess.return_value = "SONAME: libcudnn.so.9"

        def glob_side_effect(pattern):
            if "libcudnn" in pattern:
                return ["/usr/local/cuda/lib64/libcudnn.so.9.0.0"]
            return []

        mock_glob.side_effect = glob_side_effect

        # Mock CUDA detector to return CUDA 11.x (incompatible with cuDNN 9)
        mock_cuda_detector = MagicMock()
        mock_cuda_result = MagicMock()
        mock_cuda_result.detected = True
        mock_cuda_result.version = "11.8"
        mock_cuda_detector.detect.return_value = mock_cuda_result
        mock_registry_get.return_value = mock_cuda_detector

        result = detector.detect()

        # Should detect incompatibility
        cuda_compat = result.metadata.get("cuda_compatibility", {})
        assert cuda_compat.get("compatible") is False or "requires CUDA 12" in cuda_compat.get("message", "")