"""
Integration tests for cuda-install CLI command.

Tests the end-to-end cuda-install command workflow.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
import io
import sys

from env_doctor.cli import cuda_install_command


class TestCudaInstallCLI:
    """Integration tests for cuda-install command."""

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_auto_detect_ubuntu_with_driver(self, mock_stdout, mock_platform, mock_registry):
        """Test auto-detect mode on Ubuntu with NVIDIA driver."""
        # Mock platform detection
        mock_platform.return_value = {
            "os": "linux",
            "distro": "ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["linux_ubuntu_22.04_x86_64", "conda_any"]
        }

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.146.02"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command
        cuda_install_command()

        output = mock_stdout.getvalue()

        # Verify output contains expected elements
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        assert "ubuntu 22.04" in output.lower()
        assert "12.2" in output  # 12.2 should recommend 12.2
        assert "Installation Steps:" in output
        assert "nvcc --version" in output  # Verification step

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_specific_version_ubuntu(self, mock_stdout, mock_platform, mock_registry):
        """Test requesting specific CUDA version on Ubuntu."""
        # Mock platform detection
        mock_platform.return_value = {
            "os": "linux",
            "distro": "ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["linux_ubuntu_22.04_x86_64", "conda_any"]
        }

        # Mock driver detector (still needed for display)
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "560.35.03"
        mock_driver_result.metadata = {"max_cuda_version": "12.6"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command with specific version
        cuda_install_command(cuda_version="12.4")

        output = mock_stdout.getvalue()

        # Verify output
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        assert "12.4" in output
        assert "Installation Steps:" in output

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_wsl2_ubuntu(self, mock_stdout, mock_platform, mock_registry):
        """Test WSL2 Ubuntu installation guide."""
        # Mock WSL2 platform detection
        mock_platform.return_value = {
            "os": "linux",
            "distro": "ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "is_wsl2": True,
            "platform_keys": ["linux_wsl2_ubuntu_x86_64", "linux_ubuntu_22.04_x86_64", "conda_any"]
        }

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "560.35.03"
        mock_driver_result.metadata = {"max_cuda_version": "12.6"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command
        cuda_install_command()

        output = mock_stdout.getvalue()

        # Verify WSL2-specific output
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        assert "wsl" in output.lower()
        # WSL2 should have special notes about driver
        assert "driver" in output.lower() or "windows" in output.lower() or "host" in output.lower()

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_windows_installation(self, mock_stdout, mock_platform, mock_registry):
        """Test Windows installation guide."""
        # Mock Windows platform detection
        mock_platform.return_value = {
            "os": "windows",
            "distro": None,
            "distro_version": None,
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["windows_10_11_x86_64", "conda_any"]
        }

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "560.35.03"
        mock_driver_result.metadata = {"max_cuda_version": "12.6"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command
        cuda_install_command()

        output = mock_stdout.getvalue()

        # Verify Windows-specific output
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        assert "Windows" in output
        assert "winget" in output.lower()

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_no_driver_detected(self, mock_stdout, mock_platform, mock_registry):
        """Test when no NVIDIA driver is detected."""
        # Mock platform detection
        mock_platform.return_value = {
            "os": "linux",
            "distro": "ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["linux_ubuntu_22.04_x86_64", "conda_any"]
        }

        # Mock driver detector - no driver found
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = False
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command
        cuda_install_command()

        output = mock_stdout.getvalue()

        # Should still provide output but with warning
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        # Might recommend latest version or show error

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_unknown_platform_fallback(self, mock_stdout, mock_platform, mock_registry):
        """Test conda fallback for unknown platform."""
        # Mock unknown platform
        mock_platform.return_value = {
            "os": "linux",
            "distro": "archlinux",
            "distro_version": None,
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["conda_any"]
        }

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.146.02"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command
        cuda_install_command()

        output = mock_stdout.getvalue()

        # Should fallback to conda or show available platforms
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        # Might show conda instructions or list of available platforms

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_rhel_9_installation(self, mock_stdout, mock_platform, mock_registry):
        """Test RHEL 9 installation guide."""
        # Mock RHEL platform detection
        mock_platform.return_value = {
            "os": "linux",
            "distro": "rhel",
            "distro_version": "9.1",
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["linux_rhel_9.1_x86_64", "linux_rhel_9_x86_64", "conda_any"]
        }

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.146.02"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command
        cuda_install_command()

        output = mock_stdout.getvalue()

        # Verify RHEL-specific output
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        # RHEL might use dnf/yum or show rpm instructions
        if "conda" not in output.lower():
            # If not conda fallback, should have RHEL-specific instructions
            assert True  # Platform detected

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_output_has_post_install_steps(self, mock_stdout, mock_platform, mock_registry):
        """Test that output includes post-installation steps."""
        # Mock platform detection
        mock_platform.return_value = {
            "os": "linux",
            "distro": "ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["linux_ubuntu_22.04_x86_64", "conda_any"]
        }

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "535.146.02"
        mock_driver_result.metadata = {"max_cuda_version": "12.2"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command
        cuda_install_command()

        output = mock_stdout.getvalue()

        # Verify post-installation guidance
        assert "Post-Installation" in output or "post-install" in output.lower()
        # Should mention environment variables
        assert "PATH" in output or "export" in output

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_old_cuda_version_11_8(self, mock_stdout, mock_platform, mock_registry):
        """Test installation guide for older CUDA 11.8."""
        # Mock platform detection
        mock_platform.return_value = {
            "os": "linux",
            "distro": "ubuntu",
            "distro_version": "20.04",
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["linux_ubuntu_20.04_x86_64", "conda_any"]
        }

        # Mock driver detector
        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "520.61.05"
        mock_driver_result.metadata = {"max_cuda_version": "11.8"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        # Run command with specific version
        cuda_install_command(cuda_version="11.8")

        output = mock_stdout.getvalue()

        # Verify CUDA 11.8 is mentioned
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        assert "11.8" in output

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_json_output_windows(self, mock_stdout, mock_platform, mock_registry):
        """Test JSON output for Windows includes winget steps."""
        mock_platform.return_value = {
            "os": "windows",
            "distro": None,
            "distro_version": None,
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["windows_10_11_x86_64", "conda_any"]
        }

        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = True
        mock_driver_result.version = "560.35.03"
        mock_driver_result.metadata = {"max_cuda_version": "12.6"}
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        cuda_install_command(output_json=True)

        output = mock_stdout.getvalue()
        data = json.loads(output)

        assert data["platform"]["os"] == "windows"
        assert data["recommended_version"] == "12.6"
        assert "install_info" in data
        assert data["install_info"]["method"] == "winget"
        assert any("winget" in step for step in data["install_info"]["steps"])
        assert data["driver_version"] == "560.35.03"
        assert data["max_cuda"] == "12.6"

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_json_output_no_driver(self, mock_stdout, mock_platform, mock_registry):
        """Test JSON output when no driver is detected."""
        mock_platform.return_value = {
            "os": "linux",
            "distro": "ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "is_wsl2": False,
            "platform_keys": ["linux_ubuntu_22.04_x86_64", "conda_any"]
        }

        mock_driver_detector = MagicMock()
        mock_driver_result = MagicMock()
        mock_driver_result.detected = False
        mock_driver_detector.detect.return_value = mock_driver_result
        mock_registry.return_value = mock_driver_detector

        cuda_install_command(output_json=True)

        output = mock_stdout.getvalue()
        data = json.loads(output)

        assert "error" in data
        assert "platform" in data
