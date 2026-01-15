"""
Unit tests for Dockerfile validator.
"""
import pytest
import tempfile
import os
from pathlib import Path

from env_doctor.validators.dockerfile_validator import DockerfileValidator
from env_doctor.validators.models import Severity


# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"


class TestDockerfileValidator:
    """Tests for DockerfileValidator class."""

    def test_valid_dockerfile_passes(self):
        """Test that a valid Dockerfile with CUDA base image passes."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.valid"))
        result = validator.validate()

        assert result.success
        assert result.error_count == 0
        # Should have info message about GPU-enabled base
        assert result.info_count >= 1

    def test_cpu_only_base_detected(self):
        """Test that CPU-only base images are detected."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        # Should have error about CPU-only base
        errors = result.get_issues_by_severity(Severity.ERROR)
        assert any("CPU-only base image" in issue.issue for issue in errors)

    def test_missing_index_url_detected(self):
        """Test that missing --index-url in pip install is detected."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        errors = result.get_issues_by_severity(Severity.ERROR)
        assert any("missing --index-url" in issue.issue.lower() for issue in errors)

    def test_driver_installation_detected(self):
        """Test that NVIDIA driver installation is detected and flagged."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        errors = result.get_issues_by_severity(Severity.ERROR)
        assert any("must NOT be installed" in issue.issue for issue in errors)

    def test_cuda_toolkit_flagged(self):
        """Test that CUDA toolkit installation is flagged."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        warnings = result.get_issues_by_severity(Severity.WARNING)
        assert any("toolkit" in issue.issue.lower() for issue in warnings)

    def test_line_numbers_accurate(self):
        """Test that line numbers are reported accurately."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        # All issues should have line numbers > 0
        for issue in result.issues:
            assert issue.line_number > 0

    def test_multiple_issues_in_one_file(self):
        """Test that multiple issues are detected in a single Dockerfile."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        # Should have multiple errors
        assert result.error_count >= 2
        # Should have multiple issues overall
        assert len(result.issues) >= 3

    def test_dockerfile_not_found(self):
        """Test handling of non-existent Dockerfile."""
        validator = DockerfileValidator("/nonexistent/Dockerfile")
        result = validator.validate()

        assert not result.success
        assert result.error_count == 1
        assert "not found" in result.issues[0].issue.lower()

    def test_multistage_dockerfile_validates_final_stage(self):
        """Test that multi-stage Dockerfile validates final stage only."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.multistage"))
        result = validator.validate()

        # Final stage has nvidia/cuda base, so should pass
        assert result.success or result.error_count == 0

    def test_corrected_command_provided(self):
        """Test that corrected commands are provided for fixable issues."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        # Some issues should have corrected commands
        has_correction = any(issue.corrected_command is not None for issue in result.issues)
        assert has_correction

    def test_issues_sorted_by_line_number(self):
        """Test that issues are sorted by line number."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.invalid"))
        result = validator.validate()

        line_numbers = [issue.line_number for issue in result.issues if issue.line_number > 0]
        assert line_numbers == sorted(line_numbers)

    def test_cuda_version_extraction(self):
        """Test that CUDA version is extracted from base image."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.valid"))
        result = validator.validate()

        # After validation, cuda_version should be set
        assert validator.cuda_version == "12.1"

    def test_tensorflow_validation(self):
        """Test TensorFlow installation validation."""
        # Create a temporary Dockerfile with TensorFlow
        with tempfile.NamedTemporaryFile(mode='w', suffix='Dockerfile', delete=False) as f:
            f.write("""FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN pip install tensorflow
""")
            temp_path = f.name

        try:
            validator = DockerfileValidator(temp_path)
            result = validator.validate()

            # Should have warning about tensorflow GPU support
            warnings = result.get_issues_by_severity(Severity.WARNING)
            assert any("tensorflow" in issue.issue.lower() for issue in warnings)
        finally:
            os.unlink(temp_path)

    def test_cuda_version_mismatch(self):
        """Test detection of CUDA version mismatch between base and pip install."""
        # Create a Dockerfile with mismatched CUDA versions
        with tempfile.NamedTemporaryFile(mode='w', suffix='Dockerfile', delete=False) as f:
            f.write("""FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
""")
            temp_path = f.name

        try:
            validator = DockerfileValidator(temp_path)
            result = validator.validate()

            # Should have warning about version mismatch
            warnings = result.get_issues_by_severity(Severity.WARNING)
            assert any("mismatch" in issue.issue.lower() for issue in warnings)
        finally:
            os.unlink(temp_path)

    def test_line_continuation_handling(self):
        """Test that line continuations are handled correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='Dockerfile', delete=False) as f:
            f.write("""FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN pip install \\
    torch \\
    torchvision
""")
            temp_path = f.name

        try:
            validator = DockerfileValidator(temp_path)
            result = validator.validate()

            # Should detect missing --index-url even with line continuations
            errors = result.get_issues_by_severity(Severity.ERROR)
            assert any("--index-url" in issue.issue.lower() for issue in errors)
        finally:
            os.unlink(temp_path)

    def test_comments_ignored(self):
        """Test that comments in Dockerfile are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='Dockerfile', delete=False) as f:
            f.write("""FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# This is a comment with pip install torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
""")
            temp_path = f.name

        try:
            validator = DockerfileValidator(temp_path)
            result = validator.validate()

            # Should pass - comment line should not trigger validation
            assert result.error_count == 0
        finally:
            os.unlink(temp_path)


class TestDockerfileValidatorDBIntegration:
    """Tests for DB-driven validation features."""

    def test_db_driven_torch_wheel_url_suggestion(self):
        """Test that DB-verified install commands are used for torch."""
        from env_doctor.validators.compat_db import CompatibilityDB

        # Create mock DB
        mock_db_data = {
            "recommendations": {
                "12.1": {
                    "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                }
            }
        }
        mock_db = CompatibilityDB(data=mock_db_data)

        # Test with fixture
        validator = DockerfileValidator(
            str(FIXTURES_DIR / "Dockerfile.torch_missing_index_url"),
            compat_db=mock_db
        )
        result = validator.validate()

        # Should have ERROR for missing index-url
        errors = result.get_issues_by_severity(Severity.ERROR)
        assert any("missing --index-url" in issue.issue.lower() for issue in errors)

        # Corrected command should contain DB-verified command
        torch_issue = next((issue for issue in errors if "--index-url" in issue.issue.lower()), None)
        assert torch_issue is not None
        assert torch_issue.corrected_command is not None
        assert "cu121" in torch_issue.corrected_command

    def test_cpu_base_with_gpu_libs_db_recommendation(self):
        """Test CPU base image with GPU libraries gets DB-driven recommendations."""
        from env_doctor.validators.compat_db import CompatibilityDB

        # Create mock DB with torch==2.0.1 for CUDA 11.7
        mock_db_data = {
            "recommendations": {
                "11.7": {
                    "torch": "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117"
                },
                "12.1": {
                    "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                }
            }
        }
        mock_db = CompatibilityDB(data=mock_db_data)

        validator = DockerfileValidator(
            str(FIXTURES_DIR / "Dockerfile.cpu_base_with_torch"),
            compat_db=mock_db
        )
        result = validator.validate()

        # Should have ERROR about CPU-only base
        errors = result.get_issues_by_severity(Severity.ERROR)
        cpu_issue = next((issue for issue in errors if "CPU-only" in issue.issue), None)
        assert cpu_issue is not None

        # Should recommend CUDA base with DB-verified command
        assert cpu_issue.corrected_command is not None
        assert "FROM nvidia/cuda:" in cpu_issue.corrected_command
        assert "11.7" in cpu_issue.corrected_command or "12.1" in cpu_issue.corrected_command

    def test_runtime_needs_compilation_error(self):
        """Test that compilation packages with runtime base image trigger error."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.runtime_needs_compilation"))
        result = validator.validate()

        # Should have ERROR about runtime needing devel
        errors = result.get_issues_by_severity(Severity.ERROR)
        compilation_issue = next((issue for issue in errors if "compilation" in issue.issue.lower()), None)
        assert compilation_issue is not None
        assert "-devel" in compilation_issue.recommendation or "-devel" in str(compilation_issue.corrected_command or "")

    def test_pinned_version_mismatch_warning(self):
        """Test that pinned versions not in DB for CUDA trigger warning."""
        from env_doctor.validators.compat_db import CompatibilityDB

        # Create mock DB where CUDA 11.7 has torch 2.0.1, but user pins 2.5.0
        mock_db_data = {
            "recommendations": {
                "11.7": {
                    "torch": "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117"
                },
                "12.1": {
                    "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                }
            }
        }
        mock_db = CompatibilityDB(data=mock_db_data)

        validator = DockerfileValidator(
            str(FIXTURES_DIR / "Dockerfile.pinned_version_mismatch"),
            compat_db=mock_db
        )
        result = validator.validate()

        # Should have WARNING about version mismatch
        warnings = result.get_issues_by_severity(Severity.WARNING)
        mismatch_issue = next((issue for issue in warnings if "differs from DB-verified" in issue.issue), None)
        assert mismatch_issue is not None
        assert "2.5.0" in mismatch_issue.issue
        assert "11.7" in mismatch_issue.issue

    def test_multi_library_all_verified_info(self):
        """Test that multiple verified libraries show INFO summary."""
        from env_doctor.validators.compat_db import CompatibilityDB

        # Create mock DB with both torch and tensorflow verified for CUDA 12.1
        mock_db_data = {
            "recommendations": {
                "12.1": {
                    "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
                    "tensorflow": "pip install tensorflow==2.15.0"
                }
            }
        }
        mock_db = CompatibilityDB(data=mock_db_data)

        validator = DockerfileValidator(
            str(FIXTURES_DIR / "Dockerfile.multi_lib_all_verified"),
            compat_db=mock_db
        )
        result = validator.validate()

        # Should have INFO about multiple libraries
        infos = result.get_issues_by_severity(Severity.INFO)
        multi_lib_info = next((issue for issue in infos if "Multiple GPU libraries" in issue.issue), None)
        assert multi_lib_info is not None
        assert "torch" in multi_lib_info.issue.lower()
        assert "tensorflow" in multi_lib_info.issue.lower()

    def test_deprecated_tensorflow_gpu(self):
        """Test that tensorflow-gpu triggers deprecation warning."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.tensorflow_gpu_deprecated"))
        result = validator.validate()

        # Should have WARNING about deprecated package
        warnings = result.get_issues_by_severity(Severity.WARNING)
        deprecation_issue = next((issue for issue in warnings if "tensorflow-gpu" in issue.issue.lower() and "deprecated" in issue.issue.lower()), None)
        assert deprecation_issue is not None

        # Should suggest replacement
        assert "tensorflow[and-cuda]" in deprecation_issue.recommendation or "tensorflow[and-cuda]" in str(deprecation_issue.corrected_command or "")

    def test_base_image_flavor_detection(self):
        """Test that base image flavor (runtime/devel) is correctly detected."""
        validator = DockerfileValidator(str(FIXTURES_DIR / "Dockerfile.runtime_needs_compilation"))
        result = validator.validate()

        # After validation, flavor should be detected
        assert validator.base_image_flavor == "runtime"

    def test_detected_libraries_tracking(self):
        """Test that detected libraries are tracked correctly."""
        from env_doctor.validators.compat_db import CompatibilityDB

        mock_db = CompatibilityDB(data={"recommendations": {}})

        validator = DockerfileValidator(
            str(FIXTURES_DIR / "Dockerfile.multi_lib_all_verified"),
            compat_db=mock_db
        )
        result = validator.validate()

        # Should have detected both torch and tensorflow
        assert "torch" in validator.detected_libraries
        assert "tensorflow" in validator.detected_libraries
        assert validator.detected_libraries["torch"]["line_number"] > 0
        assert validator.detected_libraries["tensorflow"]["line_number"] > 0
