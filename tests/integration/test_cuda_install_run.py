"""
Integration tests for cuda-install --run command.

Tests the full CLI flow with mocked subprocess execution.
"""
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
import io
import sys

from env_doctor.cli import cuda_install_command
from env_doctor.ci import CIEnvironment


@contextmanager
def mock_geteuid(euid):
    """Context manager to mock os.geteuid, creating it if needed (Windows)."""
    had_geteuid = hasattr(os, 'geteuid')
    if not had_geteuid:
        os.geteuid = lambda: euid
        try:
            yield
        finally:
            del os.geteuid
    else:
        with patch('os.geteuid', return_value=euid):
            yield


def mock_driver_detected(max_cuda="12.6", driver_version="560.35.03"):
    """Create mock for driver detection."""
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_result.detected = True
    mock_result.version = driver_version
    mock_result.metadata = {"max_cuda_version": max_cuda}
    mock_detector.detect.return_value = mock_result
    return mock_detector


def mock_ubuntu_platform():
    return {
        "os": "linux",
        "distro": "ubuntu",
        "distro_version": "22.04",
        "arch": "x86_64",
        "is_wsl2": False,
        "platform_key": "linux_ubuntu_22.04_x86_64",
        "platform_keys": ["linux_ubuntu_22.04_x86_64", "conda_any"],
    }


def mock_windows_platform():
    return {
        "os": "windows",
        "distro": "windows",
        "distro_version": "10",
        "arch": "x86_64",
        "is_wsl2": False,
        "platform_key": "windows_10_11_x86_64",
        "platform_keys": ["windows_10_11_x86_64", "conda_any"],
    }


def make_successful_popen_mock():
    mock_process = MagicMock()
    mock_process.stdout = iter([])
    mock_process.stderr.read.return_value = ""
    mock_process.wait.return_value = None
    mock_process.returncode = 0
    return mock_process


class TestDefaultBehaviorUnchanged:
    """Verify that default (no --run) behavior is not affected."""

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_display_only_by_default(self, mock_stdout, mock_platform, mock_registry):
        """Without --run, should display instructions, not execute."""
        mock_platform.return_value = mock_ubuntu_platform()
        mock_registry.return_value = mock_driver_detected()

        cuda_install_command()

        output = mock_stdout.getvalue()
        assert "CUDA TOOLKIT INSTALLATION GUIDE" in output
        assert "Installation Steps:" in output


class TestDryRunCLI:
    """Test --dry-run flag through the CLI function."""

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('env_doctor.installer.subprocess')
    def test_dry_run_ubuntu(self, mock_subprocess, mock_platform, mock_registry, capsys):
        """Dry run on Ubuntu should show steps without executing."""
        mock_platform.return_value = mock_ubuntu_platform()
        mock_registry.return_value = mock_driver_detected()

        cuda_install_command(dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        mock_subprocess.Popen.assert_not_called()

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('env_doctor.installer.subprocess')
    def test_dry_run_windows(self, mock_subprocess, mock_platform, mock_registry, capsys):
        """Dry run on Windows should show winget step."""
        mock_platform.return_value = mock_windows_platform()
        mock_registry.return_value = mock_driver_detected()

        cuda_install_command(dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        mock_subprocess.Popen.assert_not_called()


class TestRunWithMockedSubprocess:
    """Test --run flag with mocked subprocess for Ubuntu."""

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('env_doctor.installer.subprocess.Popen')
    def test_run_success_ubuntu(self, mock_popen, mock_platform, mock_registry):
        """Successful run on Ubuntu exits with code 0."""
        mock_platform.return_value = mock_ubuntu_platform()
        mock_registry.return_value = mock_driver_detected()
        mock_popen.return_value = make_successful_popen_mock()

        with mock_geteuid(0):
            with pytest.raises(SystemExit) as exc_info:
                cuda_install_command(run=True, yes=True)

        assert exc_info.value.code == 0

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('env_doctor.installer.subprocess.Popen')
    def test_run_failure_exits_1(self, mock_popen, mock_platform, mock_registry):
        """Failed installation step exits with code 1."""
        mock_platform.return_value = mock_ubuntu_platform()
        mock_registry.return_value = mock_driver_detected()

        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.stderr.read.return_value = "E: Package not found"
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with mock_geteuid(0):
            with pytest.raises(SystemExit) as exc_info:
                cuda_install_command(run=True, yes=True)

        assert exc_info.value.code == 1


class TestRunWithJSON:
    """Test --run --json output mode."""

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('env_doctor.installer.subprocess.Popen')
    def test_json_output_with_run(self, mock_popen, mock_platform, mock_registry, capsys):
        """--run --json should output a valid JSON InstallResult."""
        mock_platform.return_value = mock_ubuntu_platform()
        mock_registry.return_value = mock_driver_detected()
        mock_popen.return_value = make_successful_popen_mock()

        with mock_geteuid(0):
            with pytest.raises(SystemExit) as exc_info:
                cuda_install_command(run=True, yes=True, output_json=True)

        assert exc_info.value.code == 0

        output = capsys.readouterr().out
        # Find the JSON block in output
        lines = output.strip().split('\n')
        json_str = ""
        brace_count = 0
        in_json = False
        for line in lines:
            if line.strip().startswith('{'):
                in_json = True
            if in_json:
                json_str += line + "\n"
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    break

        data = json.loads(json_str)
        assert data["success"] is True
        assert data["cuda_version"] == "12.6"
        assert "steps_completed" in data


class TestSpecificVersion:
    """Test --run with a specific CUDA version."""

    @patch('env_doctor.core.registry.DetectorRegistry.get')
    @patch('env_doctor.utilities.platform_detect.detect_platform')
    @patch('env_doctor.installer.subprocess')
    def test_specific_version_dry_run(self, mock_subprocess, mock_platform, mock_registry, capsys):
        """Specific version with --dry-run should work."""
        mock_platform.return_value = mock_ubuntu_platform()
        mock_registry.return_value = mock_driver_detected()

        cuda_install_command(cuda_version="12.4", dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
