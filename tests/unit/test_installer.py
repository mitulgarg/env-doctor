"""
Unit tests for the CUDA installer execution engine.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open, call, PropertyMock
from pathlib import Path
from contextlib import contextmanager

from env_doctor.installer import CudaInstaller, InstallResult, StepResult
from env_doctor.ci import CIEnvironment
from env_doctor.core.exceptions import PrivilegeError


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


def make_install_info(steps=None, post_install=None, verify="nvcc --version"):
    """Helper to create install_info dicts."""
    return {
        "method": "network_deb",
        "label": "Ubuntu 22.04 (x86_64) - Network Install",
        "steps": steps or [
            "sudo apt-get update",
            "sudo apt-get -y install cuda-toolkit-12-6",
        ],
        "post_install": post_install or [
            "export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}",
            "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}",
        ],
        "verify": verify,
    }


def make_platform_info():
    return {
        "os": "linux",
        "distro": "ubuntu",
        "distro_version": "22.04",
        "arch": "x86_64",
        "is_wsl2": False,
        "platform_key": "linux_ubuntu_22.04_x86_64",
        "platform_keys": ["linux_ubuntu_22.04_x86_64", "conda_any"],
    }


def make_ci_env(name="local", is_ci=False, method="shell_rc"):
    return CIEnvironment(name=name, is_ci=is_ci, env_persist_method=method)


def make_successful_popen_mock():
    """Create a MagicMock for subprocess.Popen that returns success."""
    mock_process = MagicMock()
    mock_process.stdout = iter([])
    mock_process.stderr.read.return_value = ""
    mock_process.wait.return_value = None
    mock_process.returncode = 0
    return mock_process


class TestDryRun:
    """Test dry run mode - no subprocess calls should be made."""

    @patch('env_doctor.installer.subprocess')
    def test_dry_run_no_subprocess(self, mock_subprocess):
        """Dry run should not call subprocess at all."""
        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            dry_run=True,
        )
        result = installer.run()

        assert result.success is True
        assert len(result.steps_completed) == 3  # 2 install + 1 verify
        mock_subprocess.Popen.assert_not_called()
        for step in result.steps_completed:
            assert step.return_code == 0

    def test_dry_run_shows_all_steps(self, capsys):
        """Dry run should print all commands."""
        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            dry_run=True,
        )
        result = installer.run()
        output = capsys.readouterr().out

        assert "[DRY RUN]" in output
        assert "sudo apt-get update" in output
        assert "sudo apt-get -y install cuda-toolkit-12-6" in output
        assert result.log_file is not None


class TestHappyPath:
    """Test successful installation flow."""

    @patch('env_doctor.installer.subprocess.Popen')
    @patch('env_doctor.installer.subprocess.run')
    def test_all_steps_succeed(self, mock_run, mock_popen):
        """All steps pass, verification passes."""
        mock_popen.return_value = make_successful_popen_mock()
        mock_run.return_value = MagicMock(returncode=0)

        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(name="generic_ci", is_ci=True, method="export_echo"),
            yes=True,
        )

        with mock_geteuid(1000):
            result = installer.run()

        assert result.success is True
        assert result.verification_passed is True
        assert len(result.steps_remaining) == 0
        assert result.error_message is None

    @patch('env_doctor.installer.subprocess.Popen')
    def test_step_failure_stops_execution(self, mock_popen):
        """If step 1 fails, step 2 should not run."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.stderr.read.return_value = "E: Unable to locate package"
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        installer = CudaInstaller(
            install_info=make_install_info(steps=[
                "sudo apt-get update",
                "sudo apt-get -y install cuda-toolkit-12-6",
                "echo done",
            ]),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            yes=True,
        )

        with mock_geteuid(0):
            result = installer.run()

        assert result.success is False
        assert len(result.steps_completed) == 1
        assert result.steps_remaining == [
            "sudo apt-get -y install cuda-toolkit-12-6",
            "echo done",
        ]
        assert "Step 1 failed" in result.error_message

    @patch('env_doctor.installer.subprocess.Popen')
    def test_verification_failure(self, mock_popen):
        """Install ok, but verification (nvcc) fails."""
        call_count = [0]

        def popen_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_process = MagicMock()
            mock_process.stdout = iter([])
            mock_process.stderr.read.return_value = ""
            mock_process.wait.return_value = None
            # Last call (verification) fails
            if call_count[0] == 3:  # 2 install steps + 1 verify
                mock_process.returncode = 1
            else:
                mock_process.returncode = 0
            return mock_process

        mock_popen.side_effect = popen_side_effect

        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            yes=True,
        )

        with mock_geteuid(0):
            result = installer.run()

        assert result.success is True  # Install succeeded
        assert result.verification_passed is False
        assert "verification failed" in result.error_message.lower()


class TestPrivilegeChecks:
    """Test privilege checking for sudo commands."""

    @patch('env_doctor.installer.subprocess.Popen')
    def test_root_user_passes(self, mock_popen):
        """Root user (euid 0) should pass privilege check."""
        mock_popen.return_value = make_successful_popen_mock()

        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            yes=True,
        )

        with mock_geteuid(0):
            result = installer.run()

        assert result.success is True

    @patch('platform.system', return_value='Linux')
    @patch('env_doctor.installer.subprocess.Popen')
    @patch('env_doctor.installer.subprocess.run')
    def test_non_root_no_sudo_in_yes_mode(self, mock_run, mock_popen, mock_system):
        """Non-root user without passwordless sudo in --yes mode should fail."""
        mock_run.return_value = MagicMock(returncode=1)

        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(name="generic_ci", is_ci=True, method="export_echo"),
            yes=True,
        )

        with mock_geteuid(1000):
            result = installer.run()

        assert result.success is False
        assert "root privileges" in result.error_message.lower() or "privilege" in result.error_message.lower()

    def test_no_sudo_needed_for_winget(self):
        """Windows winget steps don't need sudo."""
        installer = CudaInstaller(
            install_info=make_install_info(
                steps=["winget install Nvidia.CUDA --version 12.6"],
                post_install=["The installer automatically adds CUDA to PATH"],
            ),
            cuda_version="12.6",
            platform_info={
                "os": "windows",
                "distro": "windows",
                "arch": "x86_64",
                "is_wsl2": False,
                "platform_key": "windows_10_11_x86_64",
                "platform_keys": ["windows_10_11_x86_64", "conda_any"],
            },
            ci_env=make_ci_env(method="windows_setx"),
            dry_run=True,
        )
        result = installer.run()
        assert result.success is True


class TestConfirmation:
    """Test user confirmation prompt."""

    @patch('env_doctor.installer.subprocess.Popen')
    @patch('builtins.input', return_value='y')
    def test_user_confirms_yes(self, mock_input, mock_popen):
        """User typing 'y' should proceed."""
        mock_popen.return_value = make_successful_popen_mock()

        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            yes=False,
        )

        with mock_geteuid(0):
            result = installer.run()

        assert result.success is True

    @patch('builtins.input', return_value='n')
    def test_user_confirms_no(self, mock_input):
        """User typing 'n' should cancel."""
        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            yes=False,
        )

        with mock_geteuid(0):
            result = installer.run()

        assert result.success is False
        assert "cancelled" in result.error_message.lower()

    @patch('env_doctor.installer.subprocess.Popen')
    def test_yes_flag_skips_prompt(self, mock_popen):
        """--yes flag should skip the prompt entirely."""
        mock_popen.return_value = make_successful_popen_mock()

        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            yes=True,
        )

        with mock_geteuid(0), \
             patch('builtins.input') as mock_input:
            result = installer.run()

        mock_input.assert_not_called()
        assert result.success is True


class TestKeyboardInterrupt:
    """Test Ctrl+C handling during installation."""

    @patch('env_doctor.installer.subprocess.Popen')
    def test_keyboard_interrupt(self, mock_popen):
        """Ctrl+C should return partial result."""
        mock_popen.side_effect = KeyboardInterrupt()

        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            yes=True,
        )

        with mock_geteuid(0):
            result = installer.run()

        assert result.success is False
        assert "interrupt" in result.error_message.lower()


class TestLogFile:
    """Test install log file creation."""

    def test_log_file_created(self):
        """Log file should be created during dry run."""
        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            dry_run=True,
        )
        result = installer.run()

        assert result.log_file is not None
        log_path = Path(result.log_file)
        assert log_path.exists()
        content = log_path.read_text()
        assert "env-doctor cuda-install 12.6" in content
        assert "DRY RUN" in content or "Dry run" in content

    def test_log_file_has_timestamps(self):
        """Log entries should have timestamps."""
        installer = CudaInstaller(
            install_info=make_install_info(),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            dry_run=True,
        )
        result = installer.run()

        content = Path(result.log_file).read_text()
        import re
        assert re.search(r'\[\d{2}:\d{2}:\d{2}\]', content)


class TestInstallResult:
    """Test InstallResult serialization."""

    def test_to_dict(self):
        result = InstallResult(
            success=True,
            cuda_version="12.6",
            platform_key="linux_ubuntu_22.04_x86_64",
            steps_completed=[
                StepResult(
                    command="echo hello",
                    phase="install",
                    success=True,
                    return_code=0,
                    stdout="hello",
                    stderr="",
                    duration_seconds=0.1,
                ),
            ],
            verification_passed=True,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["cuda_version"] == "12.6"
        assert len(d["steps_completed"]) == 1
        assert d["steps_completed"][0]["command"] == "echo hello"
        assert d["verification_passed"] is True

    def test_to_dict_with_error(self):
        result = InstallResult(
            success=False,
            cuda_version="12.6",
            platform_key="unknown",
            error_message="Step 1 failed",
            steps_remaining=["echo step2"],
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error_message"] == "Step 1 failed"
        assert d["steps_remaining"] == ["echo step2"]


class TestEnvPersistence:
    """Test environment variable persistence methods."""

    def test_github_env_persistence(self, tmp_path):
        """Test writing env vars to $GITHUB_ENV."""
        env_file = tmp_path / "github_env"
        env_file.touch()
        path_file = tmp_path / "github_path"
        path_file.touch()

        ci_env = CIEnvironment(
            name="github_actions",
            is_ci=True,
            env_persist_method="github_env",
            github_env_file=str(env_file),
            github_path_file=str(path_file),
        )

        installer = CudaInstaller(
            install_info=make_install_info(
                steps=["echo test"],
                post_install=[
                    "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64",
                ],
            ),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=ci_env,
            yes=True,
        )

        with patch('env_doctor.installer.subprocess.Popen') as mock_popen, \
             mock_geteuid(0):
            mock_popen.return_value = make_successful_popen_mock()
            result = installer.run()

        assert "LD_LIBRARY_PATH" in result.env_vars_set
        content = env_file.read_text()
        assert "LD_LIBRARY_PATH" in content

    def test_no_export_lines_parsed(self):
        """Non-export post_install lines should be ignored."""
        installer = CudaInstaller(
            install_info=make_install_info(
                steps=["echo test"],
                post_install=[
                    "The installer automatically adds CUDA to PATH",
                    "Restart your terminal/IDE after installation",
                ],
            ),
            cuda_version="12.6",
            platform_info=make_platform_info(),
            ci_env=make_ci_env(),
            dry_run=True,
        )
        result = installer.run()
        assert result.env_vars_set == {}
