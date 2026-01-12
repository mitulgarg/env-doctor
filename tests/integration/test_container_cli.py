"""
Integration tests for container validation CLI commands.
"""
import pytest
import subprocess
import sys
from pathlib import Path


# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestDockerfileCLI:
    """Integration tests for 'env-doctor dockerfile' command."""

    def test_dockerfile_command_with_valid_file(self):
        """Test dockerfile command with a valid Dockerfile."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "dockerfile",
             str(FIXTURES_DIR / "Dockerfile.valid")],
            capture_output=True,
            text=True
        )

        # Should exit with code 0 (success)
        assert result.returncode == 0
        # Output should indicate success
        assert "No issues found" in result.stdout or "INFO" in result.stdout

    def test_dockerfile_command_with_invalid_file(self):
        """Test dockerfile command with an invalid Dockerfile."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "dockerfile",
             str(FIXTURES_DIR / "Dockerfile.invalid")],
            capture_output=True,
            text=True
        )

        # Should exit with error code
        assert result.returncode == 1
        # Output should show errors
        assert "ERROR" in result.stdout
        assert "CPU-only base image" in result.stdout or "missing --index-url" in result.stdout

    def test_dockerfile_command_default_path(self):
        """Test dockerfile command with default Dockerfile path."""
        # This will fail to find file, but should handle gracefully
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "dockerfile"],
            capture_output=True,
            text=True,
            cwd="/tmp"  # Use a directory without Dockerfile
        )

        # Should exit with error code
        assert result.returncode == 1
        # Should mention file not found
        assert "not found" in result.stdout.lower()

    def test_dockerfile_command_output_formatting(self):
        """Test that dockerfile command output is properly formatted."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "dockerfile",
             str(FIXTURES_DIR / "Dockerfile.invalid")],
            capture_output=True,
            text=True
        )

        # Check for expected formatting elements
        assert "DOCKERFILE VALIDATION" in result.stdout
        assert "SUMMARY" in result.stdout
        assert "Errors:" in result.stdout
        # Should have emoji indicators
        assert "❌" in result.stdout or "ERROR" in result.stdout


class TestDockerComposeCLI:
    """Integration tests for 'env-doctor docker-compose' command."""

    def test_compose_command_with_valid_file(self):
        """Test docker-compose command with a valid file."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "docker-compose",
             str(FIXTURES_DIR / "docker-compose.valid.yml")],
            capture_output=True,
            text=True
        )

        # Should exit with code 0 or have only warnings (no errors)
        # May have warnings about host system
        if result.returncode != 0:
            # If non-zero, should only have warnings, not errors
            assert "ERROR" not in result.stdout or "nvidia-container-toolkit" in result.stdout

    def test_compose_command_with_invalid_file(self):
        """Test docker-compose command with an invalid file."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "docker-compose",
             str(FIXTURES_DIR / "docker-compose.invalid.yml")],
            capture_output=True,
            text=True
        )

        # Should exit with error code
        assert result.returncode == 1
        # Output should show errors
        assert "ERROR" in result.stdout
        # Should mention specific issues
        has_issue = any(keyword in result.stdout for keyword in
                       ["Missing GPU", "driver must be", "Deprecated"])
        assert has_issue

    def test_compose_command_default_path(self):
        """Test docker-compose command with default path."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "docker-compose"],
            capture_output=True,
            text=True,
            cwd="/tmp"  # Use a directory without docker-compose.yml
        )

        # Should exit with error code
        assert result.returncode == 1
        # Should mention file not found
        assert "not found" in result.stdout.lower()

    def test_compose_command_output_formatting(self):
        """Test that docker-compose command output is properly formatted."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "docker-compose",
             str(FIXTURES_DIR / "docker-compose.invalid.yml")],
            capture_output=True,
            text=True
        )

        # Check for expected formatting elements
        assert "DOCKER COMPOSE VALIDATION" in result.stdout
        assert "SUMMARY" in result.stdout
        # Should have severity indicators
        has_severity = any(keyword in result.stdout for keyword in
                          ["ERROR", "WARNING", "❌", "⚠️"])
        assert has_severity


class TestCLIHelp:
    """Test CLI help messages."""

    def test_main_help_includes_new_commands(self):
        """Test that main help includes dockerfile and docker-compose commands."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "dockerfile" in result.stdout.lower()
        assert "docker-compose" in result.stdout.lower()

    def test_dockerfile_help(self):
        """Test dockerfile command help."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "dockerfile", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Dockerfile" in result.stdout
        assert "GPU" in result.stdout or "CUDA" in result.stdout

    def test_compose_help(self):
        """Test docker-compose command help."""
        result = subprocess.run(
            [sys.executable, "-m", "env_doctor.cli", "docker-compose", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "docker-compose" in result.stdout.lower()
        assert "GPU" in result.stdout or "gpu" in result.stdout
