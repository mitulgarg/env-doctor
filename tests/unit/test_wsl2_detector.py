import pytest
from unittest.mock import patch, mock_open

from env_doctor.detectors.wsl2 import WSL2Detector
from env_doctor.core import Status


class TestWSL2Detector:
    """Test suite for WSL2Detector class."""

    @patch("env_doctor.detectors.wsl2.platform.system")
    def test_wsl2_detector_can_run_on_linux(self, mock_platform):
        """Test that detector can run on Linux platform."""
        mock_platform.return_value = "Linux"
        detector = WSL2Detector()
        assert detector.can_run() is True

    @patch("env_doctor.detectors.wsl2.platform.system")
    def test_wsl2_detector_cannot_run_on_windows(self, mock_platform):
        """Test that detector cannot run on Windows platform."""
        mock_platform.return_value = "Windows"
        detector = WSL2Detector()
        assert detector.can_run() is False

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_native_linux_detection(self, mock_file):
        """Test detection of native Linux environment."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.version == "native_linux"
        assert result.status == Status.SUCCESS
        assert result.metadata["environment"] == "Native Linux"

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSL2Detector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSL2Detector, "_check_wsl2_libcuda", return_value=True)
    @patch.object(WSL2Detector, "_check_nvidia_smi", return_value=True)
    def test_wsl2_detection_basic(self, mock_nvidia_smi, mock_libcuda, mock_internal_driver):
        """Test basic WSL2 detection with successful GPU forwarding."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.version == "wsl2"
        assert result.status == Status.SUCCESS
        assert result.metadata["environment"] == "WSL2"
        assert result.metadata["gpu_forwarding"] == "enabled"

    @patch("builtins.open", mock_open(read_data="Linux version 4.4.0-microsoft"))
    def test_wsl1_detection(self):
        """Test WSL1 detection and warning status."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.version == "wsl1"
        assert result.status == Status.ERROR
        assert result.metadata["environment"] == "WSL1"
        assert "CUDA is not supported in WSL1" in result.issues[0]
        assert "Upgrade to WSL2" in result.recommendations[0]

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSL2Detector, "_check_internal_nvidia_driver", return_value=True)
    @patch.object(WSL2Detector, "_check_wsl2_libcuda", return_value=True)
    @patch.object(WSL2Detector, "_check_nvidia_smi", return_value=True)
    def test_wsl2_internal_driver_error(self, mock_nvidia_smi, mock_libcuda, mock_internal_driver):
        """Test WSL2 with internal NVIDIA driver error."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.status == Status.ERROR
        assert "NVIDIA driver installed inside WSL" in result.issues[0]
        assert "apt remove --purge nvidia-*" in result.recommendations[0]
        # Metadata should still be populated
        assert result.metadata["has_internal_driver"] is True
        assert result.metadata["has_libcuda"] is True
        assert result.metadata["nvidia_smi_works"] is True

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSL2Detector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSL2Detector, "_check_wsl2_libcuda", return_value=False)
    @patch.object(WSL2Detector, "_check_nvidia_smi", return_value=True)
    def test_wsl2_missing_libcuda(self, mock_nvidia_smi, mock_libcuda, mock_internal_driver):
        """Test WSL2 with missing libcuda error."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.status == Status.ERROR
        assert "Missing /usr/lib/wsl/lib/libcuda.so" in result.issues[0]
        assert "Reinstall NVIDIA driver on Windows" in result.recommendations[0]

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSL2Detector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSL2Detector, "_check_wsl2_libcuda", return_value=True)
    @patch.object(WSL2Detector, "_check_nvidia_smi", return_value=False)
    def test_wsl2_nvidia_smi_failure(self, mock_nvidia_smi, mock_libcuda, mock_internal_driver):
        """Test WSL2 with nvidia-smi failure error."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.status == Status.ERROR
        assert "nvidia-smi command failed" in result.issues[0]
        assert "Install NVIDIA driver on Windows" in result.recommendations[0]

    def test_read_proc_version_exception_handling(self):
        """Test _read_proc_version handles exceptions gracefully."""
        detector = WSL2Detector()
        with patch("builtins.open", side_effect=PermissionError):
            result = detector._read_proc_version()
            assert result == ""

    def test_detect_wsl2_environment_edge_cases(self):
        """Test _detect_wsl2_environment with various version strings."""
        detector = WSL2Detector()

        # Test empty version
        with patch.object(detector, "_read_proc_version", return_value=""):
            assert detector._detect_wsl2_environment() == "native_linux"

        # Test microsoft without WSL2
        with patch.object(detector, "_read_proc_version", return_value="Linux version 4.4.0-microsoft"):
            assert detector._detect_wsl2_environment() == "wsl1"

        # Test microsoft with WSL2
        with patch.object(detector, "_read_proc_version", return_value="Linux version 5.10.0-microsoft-standard-WSL2"):
            assert detector._detect_wsl2_environment() == "wsl2"

        # Test non-microsoft
        with patch.object(detector, "_read_proc_version", return_value="Linux version 5.10.0-generic"):
            assert detector._detect_wsl2_environment() == "native_linux"

    def test_check_nvidia_smi_success(self):
        """Test _check_nvidia_smi with successful command."""
        detector = WSL2Detector()
        with patch("env_doctor.detectors.wsl2.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            assert detector._check_nvidia_smi() is True

    def test_check_nvidia_smi_failure(self):
        """Test _check_nvidia_smi with failed command."""
        detector = WSL2Detector()
        with patch("env_doctor.detectors.wsl2.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            assert detector._check_nvidia_smi() is False

    def test_check_nvidia_smi_exception(self):
        """Test _check_nvidia_smi handles exceptions."""
        detector = WSL2Detector()
        with patch("env_doctor.detectors.wsl2.subprocess.run", side_effect=Exception):
            assert detector._check_nvidia_smi() is False

    def test_check_wsl2_libcuda_exists(self):
        """Test _check_wsl2_libcuda when file exists."""
        detector = WSL2Detector()
        with patch("env_doctor.detectors.wsl2.os.path.exists", return_value=True):
            assert detector._check_wsl2_libcuda() is True

    def test_check_wsl2_libcuda_missing(self):
        """Test _check_wsl2_libcuda when file is missing."""
        detector = WSL2Detector()
        with patch("env_doctor.detectors.wsl2.os.path.exists", return_value=False):
            assert detector._check_wsl2_libcuda() is False

    def test_check_internal_nvidia_driver_exists(self):
        """Test _check_internal_nvidia_driver when driver exists."""
        detector = WSL2Detector()
        with patch("env_doctor.detectors.wsl2.os.path.exists", return_value=True):
            assert detector._check_internal_nvidia_driver() is True

    def test_check_internal_nvidia_driver_missing(self):
        """Test _check_internal_nvidia_driver when driver is missing."""
        detector = WSL2Detector()
        with patch("env_doctor.detectors.wsl2.os.path.exists", return_value=False):
            assert detector._check_internal_nvidia_driver() is False


class TestWSL2DetectorEnhancedMetadata:
    """Tests for the enhanced metadata fields."""

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2 (gcc)"))
    @patch.object(WSL2Detector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSL2Detector, "_check_wsl2_libcuda", return_value=True)
    @patch.object(WSL2Detector, "_check_nvidia_smi", return_value=True)
    def test_wsl2_metadata_includes_kernel_version(self, mock_smi, mock_lib, mock_drv):
        """Test that WSL2 detection includes kernel version in metadata."""
        detector = WSL2Detector()
        result = detector.detect()

        assert "kernel_version" in result.metadata
        assert result.metadata["kernel_version"] == "5.10.0-microsoft-standard-WSL2"

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2 (gcc)"))
    @patch.object(WSL2Detector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSL2Detector, "_check_wsl2_libcuda", return_value=True)
    @patch.object(WSL2Detector, "_check_nvidia_smi", return_value=True)
    def test_wsl2_metadata_includes_check_results(self, mock_smi, mock_lib, mock_drv):
        """Test that WSL2 detection stores all check results in metadata."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.metadata["has_internal_driver"] is False
        assert result.metadata["has_libcuda"] is True
        assert result.metadata["nvidia_smi_works"] is True
        assert result.metadata["cuda_lib_path"] == "/usr/lib/wsl/lib/libcuda.so"

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2 (gcc)"))
    @patch.object(WSL2Detector, "_check_internal_nvidia_driver", return_value=True)
    @patch.object(WSL2Detector, "_check_wsl2_libcuda", return_value=False)
    @patch.object(WSL2Detector, "_check_nvidia_smi", return_value=False)
    def test_wsl2_all_checks_fail_metadata_still_populated(self, mock_smi, mock_lib, mock_drv):
        """Test that metadata is populated even when all checks fail."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.status == Status.ERROR
        assert result.metadata["has_internal_driver"] is True
        assert result.metadata["has_libcuda"] is False
        assert result.metadata["nvidia_smi_works"] is False
        # All three issues should be reported
        assert len(result.issues) == 3
        assert len(result.recommendations) == 3

    @patch("builtins.open", mock_open(read_data="Linux version 4.4.0-microsoft"))
    def test_wsl1_metadata_includes_kernel_version(self):
        """Test that WSL1 detection includes kernel version."""
        detector = WSL2Detector()
        result = detector.detect()

        assert "kernel_version" in result.metadata
        assert result.metadata["kernel_version"] == "4.4.0-microsoft"

    @patch("builtins.open", mock_open(read_data="Linux version 5.15.0-generic (gcc)"))
    def test_native_linux_metadata_includes_kernel_version(self):
        """Test that native Linux detection includes kernel version."""
        detector = WSL2Detector()
        result = detector.detect()

        assert result.version == "native_linux"
        assert "kernel_version" in result.metadata
        assert result.metadata["kernel_version"] == "5.15.0-generic"

    def test_extract_kernel_version(self):
        """Test kernel version extraction from proc version strings."""
        detector = WSL2Detector()

        assert detector._extract_kernel_version("Linux version 5.10.0-microsoft-standard-WSL2 (gcc)") == "5.10.0-microsoft-standard-WSL2"
        assert detector._extract_kernel_version("Linux version 5.15.0-generic") == "5.15.0-generic"
        assert detector._extract_kernel_version("short") == "unknown"
        assert detector._extract_kernel_version("") == "unknown"
