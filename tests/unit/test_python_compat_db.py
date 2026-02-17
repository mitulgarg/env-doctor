"""
Unit tests for Python compatibility data loading in db.py.
"""
from unittest.mock import patch, mock_open

import pytest


class TestLoadPythonCompatibility:
    """Tests for load_python_compatibility()."""

    def test_loads_valid_structure(self):
        """Should load and return valid structure from JSON file."""
        from env_doctor.db import load_python_compatibility

        data = load_python_compatibility()

        assert isinstance(data, dict)
        assert "python_version_constraints" in data
        assert "dependency_cascades" in data
        assert isinstance(data["python_version_constraints"], list)
        assert isinstance(data["dependency_cascades"], list)

    def test_constraints_have_required_fields(self):
        """Each constraint should have required fields."""
        from env_doctor.db import load_python_compatibility

        data = load_python_compatibility()
        for constraint in data["python_version_constraints"]:
            assert "library" in constraint
            assert "min_version" in constraint
            assert "max_version" in constraint
            assert "status" in constraint

    def test_cascades_have_required_fields(self):
        """Each cascade should have required fields."""
        from env_doctor.db import load_python_compatibility

        data = load_python_compatibility()
        for cascade in data["dependency_cascades"]:
            assert "root_library" in cascade
            assert "affected_dependencies" in cascade
            assert "severity" in cascade

    def test_fallback_when_file_missing(self):
        """Should return empty structure when file is missing."""
        from env_doctor.db import load_python_compatibility

        with patch("builtins.open", side_effect=FileNotFoundError):
            data = load_python_compatibility()

        assert data == {"python_version_constraints": [], "dependency_cascades": []}

    def test_module_level_constant_loaded(self):
        """PYTHON_COMPAT_DATA should be loaded at module level."""
        from env_doctor.db import PYTHON_COMPAT_DATA

        assert isinstance(PYTHON_COMPAT_DATA, dict)
        assert "python_version_constraints" in PYTHON_COMPAT_DATA

    def test_metadata_present(self):
        """Should include _metadata section."""
        from env_doctor.db import load_python_compatibility

        data = load_python_compatibility()
        assert "_metadata" in data
        assert "schema_version" in data["_metadata"]
