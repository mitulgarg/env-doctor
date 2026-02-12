"""
Unit tests for platform detection utility.

Tests platform/OS/distro/arch detection without requiring different environments.
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock
import platform as platform_module

from env_doctor.utilities.platform_detect import (
    detect_platform,
    _detect_wsl2,
    _detect_linux_distro
)


class TestDetectWSL2:
    """Test WSL2 detection."""

    def test_wsl2_detected(self):
        """Test WSL2 is detected from /proc/version."""
        proc_version = "Linux version 5.10.102.1-microsoft-standard-WSL2"

        with patch("builtins.open", mock_open(read_data=proc_version)):
            result = _detect_wsl2()

        assert result is True

    def test_wsl1_detected(self):
        """Test WSL1 is detected (has 'microsoft' but not WSL2)."""
        proc_version = "Linux version 4.4.0-19041-Microsoft"

        with patch("builtins.open", mock_open(read_data=proc_version)):
            result = _detect_wsl2()

        # WSL1 should return True (it has 'microsoft')
        assert result is True

    def test_native_linux(self):
        """Test native Linux (no 'microsoft' in /proc/version)."""
        proc_version = "Linux version 5.15.0-58-generic"

        with patch("builtins.open", mock_open(read_data=proc_version)):
            result = _detect_wsl2()

        assert result is False

    def test_proc_version_not_found(self):
        """Test when /proc/version doesn't exist (Windows/macOS)."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _detect_wsl2()

        assert result is False

    def test_proc_version_permission_denied(self):
        """Test when /proc/version can't be read."""
        with patch("builtins.open", side_effect=PermissionError):
            result = _detect_wsl2()

        assert result is False


class TestDetectLinuxDistro:
    """Test Linux distribution detection."""

    def test_ubuntu_2204(self):
        """Test Ubuntu 22.04 detection."""
        os_release = """
NAME="Ubuntu"
VERSION="22.04.1 LTS (Jammy Jellyfish)"
ID=ubuntu
ID_LIKE=debian
VERSION_ID="22.04"
"""

        with patch("builtins.open", mock_open(read_data=os_release)):
            result = _detect_linux_distro()

        assert result["id"] == "ubuntu"
        assert result["version"] == "22.04"

    def test_debian_12(self):
        """Test Debian 12 detection."""
        os_release = """
NAME="Debian GNU/Linux"
VERSION="12 (bookworm)"
ID=debian
VERSION_ID="12"
"""

        with patch("builtins.open", mock_open(read_data=os_release)):
            result = _detect_linux_distro()

        assert result["id"] == "debian"
        assert result["version"] == "12"

    def test_rhel_9(self):
        """Test RHEL 9 detection."""
        os_release = """
NAME="Red Hat Enterprise Linux"
VERSION="9.1 (Plow)"
ID="rhel"
VERSION_ID="9.1"
"""

        with patch("builtins.open", mock_open(read_data=os_release)):
            result = _detect_linux_distro()

        assert result["id"] == "rhel"
        assert result["version"] == "9.1"

    def test_fedora_39(self):
        """Test Fedora 39 detection."""
        os_release = """
NAME="Fedora Linux"
VERSION="39 (Workstation Edition)"
ID=fedora
VERSION_ID="39"
"""

        with patch("builtins.open", mock_open(read_data=os_release)):
            result = _detect_linux_distro()

        assert result["id"] == "fedora"
        assert result["version"] == "39"

    def test_os_release_not_found(self):
        """Test when /etc/os-release doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _detect_linux_distro()

        assert result["id"] == "unknown"
        assert result["version"] == "unknown"

    def test_malformed_os_release(self):
        """Test when /etc/os-release is malformed."""
        os_release = "CORRUPT DATA\nNO PROPER FORMAT"

        with patch("builtins.open", mock_open(read_data=os_release)):
            result = _detect_linux_distro()

        # Should handle gracefully - returns "unknown" default
        assert result["id"] == "unknown"
        assert result["version"] == "unknown"


class TestDetectPlatform:
    """Test full platform detection."""

    @patch('platform.system')
    @patch('platform.machine')
    @patch('env_doctor.utilities.platform_detect._detect_wsl2')
    @patch('env_doctor.utilities.platform_detect._detect_linux_distro')
    def test_ubuntu_2204_native(self, mock_distro, mock_wsl2, mock_machine, mock_system):
        """Test native Ubuntu 22.04 x86_64 detection."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        mock_wsl2.return_value = False
        mock_distro.return_value = {"id": "ubuntu", "version": "22.04"}

        result = detect_platform()

        assert result["os"] == "linux"
        assert result["distro"] == "ubuntu"
        assert result["distro_version"] == "22.04"
        assert result["arch"] == "x86_64"
        assert result["is_wsl2"] is False
        assert "linux_ubuntu_22.04_x86_64" in result["platform_keys"]
        assert "conda_any" in result["platform_keys"]

    @patch('platform.system')
    @patch('platform.machine')
    @patch('env_doctor.utilities.platform_detect._detect_wsl2')
    @patch('env_doctor.utilities.platform_detect._detect_linux_distro')
    def test_wsl2_ubuntu(self, mock_distro, mock_wsl2, mock_machine, mock_system):
        """Test WSL2 Ubuntu detection."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        mock_wsl2.return_value = True
        mock_distro.return_value = {"id": "ubuntu", "version": "22.04"}

        result = detect_platform()

        assert result["os"] == "linux"
        assert result["is_wsl2"] is True
        # WSL2-specific key should come first
        assert result["platform_keys"][0] == "linux_wsl2_ubuntu"
        assert "linux_ubuntu_22.04_x86_64" in result["platform_keys"]

    @patch('platform.system')
    @patch('platform.machine')
    def test_windows_x86_64(self, mock_machine, mock_system):
        """Test Windows 10/11 x86_64 detection."""
        mock_system.return_value = "Windows"
        mock_machine.return_value = "AMD64"

        result = detect_platform()

        assert result["os"] == "windows"
        assert result["arch"] == "x86_64"
        assert result["is_wsl2"] is False
        assert "windows_10_11_x86_64" in result["platform_keys"]
        assert "conda_any" in result["platform_keys"]

    @patch('platform.system')
    @patch('platform.machine')
    @patch('platform.mac_ver')
    def test_macos_arm64(self, mock_mac_ver, mock_machine, mock_system):
        """Test macOS ARM64 detection."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        mock_mac_ver.return_value = ("14.0", "", "")

        result = detect_platform()

        assert result["os"] == "darwin"
        assert result["arch"] == "aarch64"  # arm64 normalized to aarch64
        assert "conda_any" in result["platform_keys"]

    @patch('platform.system')
    @patch('platform.machine')
    @patch('env_doctor.utilities.platform_detect._detect_wsl2')
    @patch('env_doctor.utilities.platform_detect._detect_linux_distro')
    def test_rhel_9_aarch64(self, mock_distro, mock_wsl2, mock_machine, mock_system):
        """Test RHEL 9 aarch64 detection."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "aarch64"
        mock_wsl2.return_value = False
        mock_distro.return_value = {"id": "rhel", "version": "9.1"}

        result = detect_platform()

        assert result["os"] == "linux"
        assert result["distro"] == "rhel"
        assert result["distro_version"] == "9.1"
        assert result["arch"] == "aarch64"
        # Major version should be in platform_keys
        assert "linux_rhel_9_aarch64" in result["platform_keys"]

    @patch('platform.system')
    @patch('platform.machine')
    @patch('env_doctor.utilities.platform_detect._detect_wsl2')
    @patch('env_doctor.utilities.platform_detect._detect_linux_distro')
    def test_unknown_linux_distro(self, mock_distro, mock_wsl2, mock_machine, mock_system):
        """Test unknown Linux distro detection."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        mock_wsl2.return_value = False
        mock_distro.return_value = {"id": "unknown", "version": "unknown"}

        result = detect_platform()

        assert result["os"] == "linux"
        assert result["distro"] == "unknown"
        assert result["distro_version"] == "unknown"
        # Should still have conda fallback
        assert "conda_any" in result["platform_keys"]

    @patch('platform.system')
    @patch('platform.machine')
    def test_platform_keys_priority(self, mock_machine, mock_system):
        """Test platform_keys are in correct priority order."""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"

        with patch('env_doctor.utilities.platform_detect._detect_wsl2', return_value=True):
            with patch('env_doctor.utilities.platform_detect._detect_linux_distro',
                      return_value={"id": "ubuntu", "version": "22.04"}):
                result = detect_platform()

        keys = result["platform_keys"]
        # WSL2-specific should be first
        assert keys[0] == "linux_wsl2_ubuntu"
        # Versioned should come before major version
        assert "linux_ubuntu_22.04_x86_64" in keys
        # Conda should be last fallback
        assert keys[-1] == "conda_any"
