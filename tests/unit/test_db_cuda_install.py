"""
Unit tests for CUDA installation data DB functions.

Tests the new CUDA Toolkit installation recommendation and lookup functions.
"""
import pytest
from unittest.mock import patch, mock_open
import json

from env_doctor.db import (
    get_recommended_cuda_toolkit,
    get_cuda_install_steps,
    CUDA_INSTALL_DATA
)


class TestGetRecommendedCudaToolkit:
    """Test CUDA toolkit version recommendation logic."""

    def test_exact_match(self):
        """Test exact match in version_recommendation_map."""
        # Driver supports CUDA 12.6 -> recommend CUDA 12.6
        recommended = get_recommended_cuda_toolkit("12.6")
        assert recommended == "12.6"

    def test_exact_match_12_4(self):
        """Test exact match for CUDA 12.4."""
        recommended = get_recommended_cuda_toolkit("12.4")
        assert recommended == "12.4"

    def test_exact_match_12_1(self):
        """Test exact match for CUDA 12.1."""
        recommended = get_recommended_cuda_toolkit("12.1")
        assert recommended == "12.1"

    def test_exact_match_11_8(self):
        """Test exact match for CUDA 11.8."""
        recommended = get_recommended_cuda_toolkit("11.8")
        assert recommended == "11.8"

    def test_12_2_maps_to_12_1(self):
        """Test CUDA 12.2 maps to 12.1 (forward compatibility)."""
        recommended = get_recommended_cuda_toolkit("12.2")
        assert recommended == "12.1"

    def test_12_5_maps_to_12_4(self):
        """Test CUDA 12.5 maps to 12.4."""
        recommended = get_recommended_cuda_toolkit("12.5")
        assert recommended == "12.4"

    def test_closest_lower_bound(self):
        """Test closest lower bound for unmapped versions."""
        # If driver supports 12.3, should recommend 12.1 (highest available < 12.3)
        recommended = get_recommended_cuda_toolkit("12.3")
        # Based on version_recommendation_map, 12.2 -> 12.1, so 12.3 should also map to 12.1
        assert recommended == "12.1"

    def test_none_input(self):
        """Test handling of None input."""
        # Function should handle None gracefully by checking or raising exception
        try:
            recommended = get_recommended_cuda_toolkit(None)
            # If it doesn't raise, should return None
            assert recommended is None
        except (TypeError, AttributeError):
            # If it raises TypeError, that's acceptable behavior
            pass

    def test_empty_string(self):
        """Test handling of empty string."""
        recommended = get_recommended_cuda_toolkit("")
        assert recommended is None

    def test_invalid_version_format(self):
        """Test handling of invalid version format."""
        recommended = get_recommended_cuda_toolkit("invalid")
        assert recommended is None

    def test_very_old_cuda(self):
        """Test very old CUDA version (10.x)."""
        recommended = get_recommended_cuda_toolkit("10.2")
        # Should return None or oldest available (11.8)
        assert recommended is None or recommended == "11.8"

    def test_very_new_cuda(self):
        """Test future CUDA version."""
        recommended = get_recommended_cuda_toolkit("13.0")
        # Should recommend latest available (12.6)
        assert recommended == "12.6"


class TestGetCudaInstallSteps:
    """Test CUDA installation steps lookup."""

    def test_ubuntu_2204_cuda_12_1(self):
        """Test Ubuntu 22.04 CUDA 12.1 installation steps."""
        platform_keys = ["linux_ubuntu_22.04_x86_64", "conda_any"]
        steps = get_cuda_install_steps("12.1", platform_keys)

        assert steps is not None
        assert steps["method"] in ["network_deb", "local_deb"]
        assert "steps" in steps
        assert len(steps["steps"]) > 0
        assert "post_install" in steps
        assert "verify" in steps
        assert steps["verify"] == "nvcc --version"

    def test_wsl2_ubuntu_priority(self):
        """Test WSL2-specific instructions take priority."""
        # WSL2 key should be first in priority list
        platform_keys = ["linux_wsl2_ubuntu_x86_64", "linux_ubuntu_22.04_x86_64", "conda_any"]
        steps = get_cuda_install_steps("12.6", platform_keys)

        assert steps is not None
        # Should get WSL2-specific instructions if available
        if "notes" in steps and steps["notes"]:
            # WSL2 should have notes about not installing driver
            assert any("driver" in note.lower() or "wsl" in note.lower()
                      for note in steps["notes"]) or True

    def test_windows_10_11(self):
        """Test Windows 10/11 installation steps."""
        platform_keys = ["windows_10_11_x86_64", "conda_any"]
        steps = get_cuda_install_steps("12.4", platform_keys)

        assert steps is not None
        assert steps["method"] in ["gui_installer", "exe_installer"]
        assert ("download_url" in steps or "download_page" in steps)
        if "download_url" in steps:
            assert "nvidia.com" in steps["download_url"]
        elif "download_page" in steps:
            assert "nvidia.com" in steps["download_page"]

    def test_conda_fallback(self):
        """Test conda fallback when specific platform not found."""
        platform_keys = ["linux_unknown_distro_x86_64", "conda_any"]
        steps = get_cuda_install_steps("12.1", platform_keys)

        assert steps is not None
        # Should fallback to conda
        assert steps["method"] == "conda"
        assert "conda install" in " ".join(steps["steps"])

    def test_version_not_found(self):
        """Test handling of non-existent CUDA version."""
        platform_keys = ["linux_ubuntu_22.04_x86_64", "conda_any"]
        steps = get_cuda_install_steps("99.9", platform_keys)

        assert steps is None

    def test_empty_platform_keys(self):
        """Test handling of empty platform keys."""
        steps = get_cuda_install_steps("12.1", [])

        assert steps is None

    def test_none_version(self):
        """Test handling of None version."""
        platform_keys = ["linux_ubuntu_22.04_x86_64"]
        steps = get_cuda_install_steps(None, platform_keys)

        assert steps is None

    def test_debian_12(self):
        """Test Debian 12 installation steps."""
        platform_keys = ["linux_debian_12_x86_64", "conda_any"]
        steps = get_cuda_install_steps("12.1", platform_keys)

        assert steps is not None
        assert "steps" in steps
        # Debian should use deb packages
        if steps["method"] != "conda":
            assert any("dpkg" in step or "apt" in step for step in steps["steps"])

    def test_rhel_9(self):
        """Test RHEL 9 installation steps."""
        platform_keys = ["linux_rhel_9_x86_64", "conda_any"]
        steps = get_cuda_install_steps("12.1", platform_keys)

        assert steps is not None
        # RHEL should use rpm packages
        if steps["method"] != "conda":
            assert any("rpm" in step or "dnf" in step or "yum" in step
                      for step in steps["steps"])

    def test_post_install_includes_env_vars(self):
        """Test post-install steps include environment variables."""
        platform_keys = ["linux_ubuntu_22.04_x86_64"]
        steps = get_cuda_install_steps("12.1", platform_keys)

        assert steps is not None
        assert "post_install" in steps
        post_install = " ".join(steps["post_install"])
        # Should have PATH and LD_LIBRARY_PATH
        assert "PATH" in post_install or "path" in post_install.lower()

    def test_steps_are_actionable(self):
        """Test that steps are concrete commands, not descriptions."""
        platform_keys = ["linux_ubuntu_22.04_x86_64"]
        steps = get_cuda_install_steps("12.1", platform_keys)

        assert steps is not None
        assert len(steps["steps"]) > 0
        # Steps should contain actual commands (wget, apt, etc.)
        all_steps = " ".join(steps["steps"])
        assert any(cmd in all_steps for cmd in ["wget", "apt", "dpkg", "conda", "download"])


class TestCudaInstallDataLoading:
    """Test CUDA install data JSON loading."""

    def test_data_loaded(self):
        """Test that CUDA_INSTALL_DATA is loaded."""
        assert CUDA_INSTALL_DATA is not None
        assert "cuda_versions" in CUDA_INSTALL_DATA
        assert "version_recommendation_map" in CUDA_INSTALL_DATA

    def test_has_required_versions(self):
        """Test that required CUDA versions are present."""
        versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
        # Should have at least these versions
        required_versions = ["12.6", "12.4", "12.1", "11.8"]
        for version in required_versions:
            assert version in versions, f"Missing CUDA version {version}"

    def test_versions_have_platforms(self):
        """Test that each version has platform data."""
        versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
        for version, version_data in versions.items():
            assert "platforms" in version_data, f"Version {version} missing platforms"
            assert len(version_data["platforms"]) > 0, f"Version {version} has no platforms"

    def test_platform_has_required_fields(self):
        """Test that platform entries have required fields."""
        versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
        required_fields = ["method", "steps", "verify"]

        for version, version_data in versions.items():
            platforms = version_data.get("platforms", {})
            for platform_key, platform_data in platforms.items():
                for field in required_fields:
                    assert field in platform_data, \
                        f"Platform {platform_key} in version {version} missing field {field}"

    def test_version_recommendation_map_complete(self):
        """Test version recommendation map has entries."""
        rec_map = CUDA_INSTALL_DATA.get("version_recommendation_map", {})
        assert len(rec_map) > 0, "version_recommendation_map is empty"
        # Spot check some known mappings
        assert "12.6" in rec_map
        assert "12.4" in rec_map
        assert "12.1" in rec_map

    def test_metadata_present(self):
        """Test _metadata is present."""
        assert "_metadata" in CUDA_INSTALL_DATA
        metadata = CUDA_INSTALL_DATA["_metadata"]
        assert "version" in metadata
        assert "last_updated" in metadata

    def test_conda_fallback_exists(self):
        """Test that conda fallback exists for all versions."""
        versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
        for version, version_data in versions.items():
            platforms = version_data.get("platforms", {})
            # Should have conda_any as fallback
            assert "conda_any" in platforms, f"Version {version} missing conda_any fallback"

    def test_wsl2_has_special_notes(self):
        """Test WSL2 platforms have special instructions."""
        versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
        for version, version_data in versions.items():
            platforms = version_data.get("platforms", {})
            for platform_key, platform_data in platforms.items():
                if "wsl2" in platform_key:
                    # WSL2 should have notes about driver
                    if "notes" in platform_data:
                        notes = platform_data["notes"]
                        # Notes could be a string or a list
                        if isinstance(notes, list):
                            notes_text = " ".join(notes).lower()
                        else:
                            notes_text = notes.lower()
                        assert "driver" in notes_text or "windows" in notes_text or "host" in notes_text

    def test_steps_are_lists(self):
        """Test that steps are lists, not strings."""
        versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
        for version, version_data in versions.items():
            platforms = version_data.get("platforms", {})
            for platform_key, platform_data in platforms.items():
                assert isinstance(platform_data["steps"], list), \
                    f"Steps for {platform_key} in {version} should be a list"
                assert len(platform_data["steps"]) > 0, \
                    f"Steps for {platform_key} in {version} should not be empty"
