import pytest
from unittest.mock import patch, mock_open

from env_doctor.detectors.wsl import WSLDetector
from env_doctor.core import Status


class TestWSLDetector:
    """Test suite for WSLDetector class."""

    @patch("env_doctor.detectors.wsl.platform.system")
    def test_wsl_detector_can_run_on_linux(self, mock_platform):
        """Test that detector can run on Linux platform."""
        mock_platform.return_value = "Linux"
        detector = WSLDetector()
        assert detector.can_run() is True

    @patch("env_doctor.detectors.wsl.platform.system")
    def test_wsl_detector_cannot_run_on_windows(self, mock_platform):
        """Test that detector cannot run on Windows platform."""
        mock_platform.return_value = "Windows"
        detector = WSLDetector()
        assert detector.can_run() is False

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_native_linux_detection(self, mock_file):
        """Test detection of native Linux environment."""
        detector = WSLDetector()
        result = detector.detect()
        
        assert result.version == "native_linux"
        assert result.status == Status.SUCCESS
        assert result.metadata["environment"] == "Native Linux"

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSLDetector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSLDetector, "_check_wsl_libcuda", return_value=True)
    @patch.object(WSLDetector, "_check_nvidia_smi", return_value=True)
    def test_wsl2_detection_basic(self, mock_nvidia_smi, mock_libcuda, mock_internal_driver):
        """Test basic WSL2 detection with successful GPU forwarding."""
        detector = WSLDetector()
        result = detector.detect()
        
        assert result.version == "wsl2"
        assert result.status == Status.SUCCESS
        assert result.metadata["environment"] == "WSL2"
        assert result.metadata["gpu_forwarding"] == "enabled"

    @patch("builtins.open", mock_open(read_data="Linux version 4.4.0-microsoft"))
    def test_wsl1_detection(self):
        """Test WSL1 detection and warning status."""
        detector = WSLDetector()
        result = detector.detect()
        
        assert result.version == "wsl1"
        assert result.status == Status.WARNING
        assert result.metadata["environment"] == "WSL1"
        assert "WSL1 detected" in result.issues[0]
        assert "Upgrade to WSL2" in result.recommendations[0]

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSLDetector, "_check_internal_nvidia_driver", return_value=True)
    def test_wsl2_internal_driver_error(self, mock_internal_driver):
        """Test WSL2 with internal NVIDIA driver error."""
        detector = WSLDetector()
        result = detector.detect()
        
        assert result.status == Status.ERROR
        assert "NVIDIA driver installed inside WSL" in result.issues[0]
        assert "apt remove --purge nvidia-*" in result.recommendations[0]

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSLDetector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSLDetector, "_check_wsl_libcuda", return_value=False)
    def test_wsl2_missing_libcuda(self, mock_libcuda, mock_internal_driver):
        """Test WSL2 with missing libcuda error."""
        detector = WSLDetector()
        result = detector.detect()
        
        assert result.status == Status.ERROR
        assert "Missing /usr/lib/wsl/lib/libcuda.so" in result.issues[0]
        assert "Reinstall NVIDIA driver on Windows" in result.recommendations[0]

    @patch("builtins.open", mock_open(read_data="Linux version 5.10.0-microsoft-standard-WSL2"))
    @patch.object(WSLDetector, "_check_internal_nvidia_driver", return_value=False)
    @patch.object(WSLDetector, "_check_wsl_libcuda", return_value=True)
    @patch.object(WSLDetector, "_check_nvidia_smi", return_value=False)
    def test_wsl2_nvidia_smi_failure(self, mock_nvidia_smi, mock_libcuda, mock_internal_driver):
        """Test WSL2 with nvidia-smi failure error."""
        detector = WSLDetector()
        result = detector.detect()
        
        assert result.status == Status.ERROR
        assert "nvidia-smi command failed" in result.issues[0]
        assert "Install NVIDIA driver on Windows" in result.recommendations[0]

    def test_read_proc_version_exception_handling(self):
        """Test _read_proc_version handles exceptions gracefully."""
        detector = WSLDetector()
        with patch("builtins.open", side_effect=PermissionError):
            result = detector._read_proc_version()
            assert result == ""

    def test_detect_wsl_type_edge_cases(self):
        """Test _detect_wsl_type with various version strings."""
        detector = WSLDetector()
        
        # Test empty version
        with patch.object(detector, "_read_proc_version", return_value=""):
            assert detector._detect_wsl_type() == "native_linux"
        
        # Test microsoft without WSL2
        with patch.object(detector, "_read_proc_version", return_value="Linux version 4.4.0-microsoft"):
            assert detector._detect_wsl_type() == "wsl1"
        
        # Test microsoft with WSL2
        with patch.object(detector, "_read_proc_version", return_value="Linux version 5.10.0-microsoft-standard-WSL2"):
            assert detector._detect_wsl_type() == "wsl2"
        
        # Test non-microsoft
        with patch.object(detector, "_read_proc_version", return_value="Linux version 5.10.0-generic"):
            assert detector._detect_wsl_type() == "native_linux"

    def test_check_nvidia_smi_success(self):
        """Test _check_nvidia_smi with successful command."""
        detector = WSLDetector()
        with patch("env_doctor.detectors.wsl.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            assert detector._check_nvidia_smi() is True

    def test_check_nvidia_smi_failure(self):
        """Test _check_nvidia_smi with failed command."""
        detector = WSLDetector()
        with patch("env_doctor.detectors.wsl.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            assert detector._check_nvidia_smi() is False

    def test_check_nvidia_smi_exception(self):
        """Test _check_nvidia_smi handles exceptions."""
        detector = WSLDetector()
        with patch("env_doctor.detectors.wsl.subprocess.run", side_effect=Exception):
            assert detector._check_nvidia_smi() is False

    def test_check_wsl_libcuda_exists(self):
        """Test _check_wsl_libcuda when file exists."""
        detector = WSLDetector()
        with patch("env_doctor.detectors.wsl.os.path.exists", return_value=True):
            assert detector._check_wsl_libcuda() is True

    def test_check_wsl_libcuda_missing(self):
        """Test _check_wsl_libcuda when file is missing."""
        detector = WSLDetector()
        with patch("env_doctor.detectors.wsl.os.path.exists", return_value=False):
            assert detector._check_wsl_libcuda() is False

    def test_check_internal_nvidia_driver_exists(self):
        """Test _check_internal_nvidia_driver when driver exists."""
        detector = WSLDetector()
        with patch("env_doctor.detectors.wsl.os.path.exists", return_value=True):
            assert detector._check_internal_nvidia_driver() is True

    def test_check_internal_nvidia_driver_missing(self):
        """Test _check_internal_nvidia_driver when driver is missing."""
        detector = WSLDetector()
        with patch("env_doctor.detectors.wsl.os.path.exists", return_value=False):
            assert detector._check_internal_nvidia_driver() is False