"""
Unit tests for PythonCompatDetector.

Tests cover conflict detection, cascade detection, edge cases,
and cross-platform compatibility.
"""
import sys
from unittest.mock import patch, MagicMock

import pytest

from env_doctor.core.detector import Status


SAMPLE_COMPAT_DATA = {
    "_metadata": {"last_verified": "2026-01-01", "schema_version": "1.0"},
    "python_version_constraints": [
        {
            "library": "tensorflow",
            "import_name": "tensorflow",
            "min_version": "3.9",
            "max_version": "3.12",
            "status": "active",
            "notes": "TF requires 3.9-3.12",
        },
        {
            "library": "torch",
            "import_name": "torch",
            "min_version": "3.9",
            "max_version": "3.12",
            "status": "active",
            "notes": "PyTorch requires 3.9-3.12",
        },
        {
            "library": "old-lib",
            "import_name": "old_lib",
            "min_version": "3.7",
            "max_version": "3.9",
            "status": "active",
            "notes": "Only supports up to 3.9",
        },
    ],
    "dependency_cascades": [
        {
            "root_library": "tensorflow",
            "affected_dependencies": ["keras", "tensorboard"],
            "severity": "high",
            "description": "TF constraint cascades to keras",
        },
        {
            "root_library": "torch",
            "affected_dependencies": ["torchvision", "torchaudio"],
            "severity": "high",
            "description": "PyTorch constraint affects ecosystem",
        },
    ],
}


def _make_detector(compat_data=None, installed_libs=None):
    """Create a PythonCompatDetector with mocked data and installed libs."""
    if compat_data is None:
        compat_data = SAMPLE_COMPAT_DATA
    if installed_libs is None:
        installed_libs = set()

    def fake_is_installed(import_name):
        return import_name in installed_libs

    with patch("env_doctor.detectors.python_compat.PYTHON_COMPAT_DATA", compat_data), \
         patch("env_doctor.detectors.python_compat._is_library_installed", side_effect=fake_is_installed):
        from env_doctor.detectors.python_compat import PythonCompatDetector
        detector = PythonCompatDetector()
        result = detector.detect()
    return result


class TestPythonCompatDetector:
    """Tests for PythonCompatDetector."""

    def test_no_libraries_installed(self):
        """When no constrained libraries are installed, should return SUCCESS."""
        result = _make_detector(installed_libs=set())
        assert result.status == Status.SUCCESS
        assert result.component == "python_compat"
        assert len(result.metadata["conflicts"]) == 0

    def test_library_within_range(self):
        """Installed library within Python version range should be SUCCESS."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=11, micro=0)
            result = _make_detector(installed_libs={"tensorflow"})
        assert result.status == Status.SUCCESS
        assert len(result.metadata["conflicts"]) == 0

    def test_library_above_maximum(self):
        """Python version above library's max should produce ERROR."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=13, micro=0)
            result = _make_detector(installed_libs={"tensorflow"})
        assert result.status == Status.ERROR
        assert len(result.metadata["conflicts"]) == 1
        assert result.metadata["conflicts"][0]["type"] == "above_maximum"
        assert "tensorflow" in result.metadata["conflicts"][0]["message"]

    def test_library_below_minimum(self):
        """Python version below library's min should produce ERROR."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=8, micro=0)
            result = _make_detector(installed_libs={"tensorflow"})
        assert result.status == Status.ERROR
        assert len(result.metadata["conflicts"]) == 1
        assert result.metadata["conflicts"][0]["type"] == "below_minimum"

    def test_multiple_conflicts(self):
        """Multiple libraries with conflicts should all be reported."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=13, micro=0)
            result = _make_detector(installed_libs={"tensorflow", "torch"})
        assert result.status == Status.ERROR
        assert len(result.metadata["conflicts"]) == 2

    def test_cascade_detection(self):
        """Conflicting library should trigger cascade reporting."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=13, micro=0)
            result = _make_detector(installed_libs={"tensorflow"})
        cascades = result.metadata["cascades"]
        assert len(cascades) == 1
        assert cascades[0]["root_library"] == "tensorflow"
        assert "keras" in cascades[0]["affected_dependencies"]

    def test_no_cascade_when_no_conflict(self):
        """No cascade should be reported when library is compatible."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=11, micro=0)
            result = _make_detector(installed_libs={"tensorflow"})
        assert len(result.metadata["cascades"]) == 0

    def test_recommendations_for_above_max(self):
        """Should recommend downgrading Python when above max."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=13, micro=0)
            result = _make_detector(installed_libs={"tensorflow"})
        assert any("3.12" in rec for rec in result.recommendations)

    def test_recommendations_for_below_min(self):
        """Should recommend upgrading Python when below min."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=8, micro=0)
            result = _make_detector(installed_libs={"tensorflow"})
        assert any("3.9" in rec for rec in result.recommendations)

    def test_empty_constraints(self):
        """Empty constraints list should return SUCCESS."""
        data = {"python_version_constraints": [], "dependency_cascades": []}
        result = _make_detector(compat_data=data, installed_libs=set())
        assert result.status == Status.SUCCESS

    def test_version_string_in_result(self):
        """Result should contain current Python version."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=11, micro=5)
            result = _make_detector(installed_libs=set())
        assert result.version == "3.11"
        assert result.metadata["python_full_version"] == "3.11.5"

    def test_inactive_constraint_ignored(self):
        """Constraints with status != 'active' should be skipped."""
        data = {
            "python_version_constraints": [
                {
                    "library": "old-pkg",
                    "import_name": "old_pkg",
                    "min_version": "3.6",
                    "max_version": "3.8",
                    "status": "deprecated",
                    "notes": "No longer maintained",
                },
            ],
            "dependency_cascades": [],
        }
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=12, micro=0)
            result = _make_detector(compat_data=data, installed_libs={"old_pkg"})
        assert result.status == Status.SUCCESS

    def test_can_run_always_true(self):
        """PythonCompatDetector.can_run() should always return True."""
        from env_doctor.detectors.python_compat import PythonCompatDetector
        detector = PythonCompatDetector()
        assert detector.can_run() is True

    def test_to_dict_serializable(self):
        """Result should be JSON-serializable via to_dict()."""
        import json
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=13, micro=0)
            result = _make_detector(installed_libs={"tensorflow", "torch"})
        data = result.to_dict()
        # Should not raise
        json.dumps(data)
        assert data["component"] == "python_compat"
        assert data["status"] == "error"

    def test_constraints_checked_count(self):
        """Should correctly count how many installed libraries were checked."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            mock_sys.version_info = MagicMock(major=3, minor=11, micro=0)
            result = _make_detector(installed_libs={"tensorflow", "torch"})
        assert result.metadata["constraints_checked"] == 2

    def test_mixed_above_and_below(self):
        """One lib below min, another above max."""
        with patch("env_doctor.detectors.python_compat.sys") as mock_sys:
            # Python 3.10 - within tensorflow range, above old-lib max
            mock_sys.version_info = MagicMock(major=3, minor=10, micro=0)
            result = _make_detector(installed_libs={"tensorflow", "old_lib"})
        assert result.status == Status.ERROR
        assert len(result.metadata["conflicts"]) == 1
        assert result.metadata["conflicts"][0]["library"] == "old-lib"
