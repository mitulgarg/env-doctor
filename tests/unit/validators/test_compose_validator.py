"""
Unit tests for Docker Compose validator.
"""
import pytest
import tempfile
import os
from pathlib import Path

from env_doctor.validators.compose_validator import ComposeValidator
from env_doctor.validators.models import Severity

# Check if PyYAML is available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestComposeValidator:
    """Tests for ComposeValidator class."""

    def test_valid_compose_passes(self):
        """Test that a valid docker-compose.yml with proper GPU config passes."""
        validator = ComposeValidator(str(FIXTURES_DIR / "docker-compose.valid.yml"))
        result = validator.validate()

        # Should have no errors
        assert result.error_count == 0
        # May have warnings about host system, but no configuration errors
        config_errors = [issue for issue in result.issues
                        if "Service" in issue.issue and issue.severity == Severity.ERROR]
        assert len(config_errors) == 0

    def test_missing_gpu_config_detected(self):
        """Test that missing GPU configuration is detected."""
        validator = ComposeValidator(str(FIXTURES_DIR / "docker-compose.invalid.yml"))
        result = validator.validate()

        errors = result.get_issues_by_severity(Severity.ERROR)
        assert any("Missing GPU device configuration" in issue.issue for issue in errors)

    def test_wrong_driver_detected(self):
        """Test that wrong GPU driver is detected."""
        validator = ComposeValidator(str(FIXTURES_DIR / "docker-compose.invalid.yml"))
        result = validator.validate()

        errors = result.get_issues_by_severity(Severity.ERROR)
        assert any("driver must be 'nvidia'" in issue.issue for issue in errors)

    def test_deprecated_runtime_flagged(self):
        """Test that deprecated 'runtime: nvidia' syntax is flagged."""
        validator = ComposeValidator(str(FIXTURES_DIR / "docker-compose.invalid.yml"))
        result = validator.validate()

        warnings = result.get_issues_by_severity(Severity.WARNING)
        assert any("Deprecated 'runtime: nvidia'" in issue.issue for issue in warnings)

    def test_multiple_services_validated(self):
        """Test that multiple services are validated."""
        validator = ComposeValidator(str(FIXTURES_DIR / "docker-compose.invalid.yml"))
        result = validator.validate()

        # Should have issues for multiple services
        service_issues = [issue for issue in result.issues if "Service" in issue.issue]
        unique_services = set()
        for issue in service_issues:
            # Extract service name from issue message
            if "Service '" in issue.issue:
                service_name = issue.issue.split("'")[1]
                unique_services.add(service_name)

        assert len(unique_services) >= 2

    def test_compose_file_not_found(self):
        """Test handling of non-existent compose file."""
        validator = ComposeValidator("/nonexistent/docker-compose.yml")
        result = validator.validate()

        assert not result.success
        assert result.error_count == 1
        assert "not found" in result.issues[0].issue.lower()

    def test_invalid_yaml_handled(self):
        """Test handling of invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
services:
  test:
    image: test
    invalid yaml: [unclosed
""")
            temp_path = f.name

        try:
            validator = ComposeValidator(temp_path)
            result = validator.validate()

            assert not result.success
            assert result.error_count >= 1
            assert any("yaml" in issue.issue.lower() for issue in result.issues)
        finally:
            os.unlink(temp_path)

    def test_corrected_command_provided(self):
        """Test that corrected GPU config is provided."""
        validator = ComposeValidator(str(FIXTURES_DIR / "docker-compose.invalid.yml"))
        result = validator.validate()

        # Some issues should have corrected commands
        has_correction = any(issue.corrected_command is not None for issue in result.issues)
        assert has_correction

    def test_non_gpu_services_skipped(self):
        """Test that services without GPU indicators are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
  database:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
""")
            temp_path = f.name

        try:
            validator = ComposeValidator(temp_path)
            result = validator.validate()

            # Should have no service-specific errors (may have host warnings)
            service_errors = [issue for issue in result.issues
                            if "Service" in issue.issue and issue.severity == Severity.ERROR]
            assert len(service_errors) == 0
        finally:
            os.unlink(temp_path)

    def test_gpu_capability_validation(self):
        """Test that missing 'gpu' in capabilities is detected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  gpu-app:
    image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [compute]
""")
            temp_path = f.name

        try:
            validator = ComposeValidator(temp_path)
            result = validator.validate()

            errors = result.get_issues_by_severity(Severity.ERROR)
            assert any("Missing 'gpu' in device capabilities" in issue.issue for issue in errors)
        finally:
            os.unlink(temp_path)

    def test_multi_service_gpu_warning(self):
        """Test that multiple GPU services trigger a warning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  train:
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  inference:
    image: tensorflow/tensorflow:latest-gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
""")
            temp_path = f.name

        try:
            validator = ComposeValidator(temp_path)
            result = validator.validate()

            warnings = result.get_issues_by_severity(Severity.WARNING)
            assert any("Multiple services use GPU" in issue.issue for issue in warnings)
        finally:
            os.unlink(temp_path)

    def test_v2_format_supported(self):
        """Test that Compose v2 format is supported."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '2.3'
services:
  gpu-app:
    image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
    runtime: nvidia
""")
            temp_path = f.name

        try:
            validator = ComposeValidator(temp_path)
            result = validator.validate()

            # Should parse successfully and detect deprecated runtime
            warnings = result.get_issues_by_severity(Severity.WARNING)
            assert any("runtime" in issue.issue.lower() for issue in warnings)
        finally:
            os.unlink(temp_path)

    def test_empty_services(self):
        """Test handling of compose file with no services."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services: {}
""")
            temp_path = f.name

        try:
            validator = ComposeValidator(temp_path)
            result = validator.validate()

            warnings = result.get_issues_by_severity(Severity.WARNING)
            assert any("No services found" in issue.issue for issue in warnings)
        finally:
            os.unlink(temp_path)
