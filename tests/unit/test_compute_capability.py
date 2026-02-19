"""
Unit tests for compute capability utilities.

Tests SM lookup, architecture name lookup, and arch list compatibility checking.
"""
import pytest

from env_doctor.detectors.compute_capability import (
    get_sm_for_compute_capability,
    get_arch_name,
    is_sm_in_arch_list,
)


class TestGetSmForComputeCapability:
    """Tests for get_sm_for_compute_capability()."""

    def test_known_cc_ampere(self):
        assert get_sm_for_compute_capability("8.6") == "sm_86"

    def test_known_cc_ada_lovelace(self):
        assert get_sm_for_compute_capability("8.9") == "sm_89"

    def test_known_cc_hopper(self):
        assert get_sm_for_compute_capability("9.0") == "sm_90"

    def test_known_cc_blackwell(self):
        assert get_sm_for_compute_capability("12.0") == "sm_120"

    def test_known_cc_blackwell_consumer(self):
        assert get_sm_for_compute_capability("10.0") == "sm_100"

    def test_unknown_cc_fallback(self):
        """Unknown CC should generate SM from the version string."""
        result = get_sm_for_compute_capability("15.3")
        assert result == "sm_153"

    def test_kepler(self):
        assert get_sm_for_compute_capability("3.5") == "sm_35"


class TestGetArchName:
    """Tests for get_arch_name()."""

    def test_known_arch_ampere(self):
        assert get_arch_name("8.0") == "Ampere"

    def test_known_arch_ada_lovelace(self):
        assert get_arch_name("8.9") == "Ada Lovelace"

    def test_known_arch_hopper(self):
        assert get_arch_name("9.0") == "Hopper"

    def test_known_arch_blackwell(self):
        assert get_arch_name("12.0") == "Blackwell"

    def test_unknown_arch(self):
        assert get_arch_name("99.9") == "Unknown"


class TestIsSmInArchList:
    """Tests for is_sm_in_arch_list()."""

    def test_direct_match(self):
        arch_list = ["sm_50", "sm_60", "sm_70", "sm_80", "sm_89"]
        assert is_sm_in_arch_list("sm_89", arch_list) is True

    def test_no_match(self):
        arch_list = ["sm_50", "sm_60", "sm_70", "sm_80"]
        assert is_sm_in_arch_list("sm_120", arch_list) is False

    def test_variant_match_sm90a(self):
        """sm_90a should cover sm_90."""
        arch_list = ["sm_50", "sm_60", "sm_70", "sm_80", "sm_90a"]
        assert is_sm_in_arch_list("sm_90", arch_list) is True

    def test_ptx_forward_compat(self):
        """compute_90 should cover sm_120 via PTX JIT."""
        arch_list = ["sm_50", "sm_60", "sm_70", "sm_80", "sm_90", "compute_90"]
        assert is_sm_in_arch_list("sm_120", arch_list) is True

    def test_ptx_forward_compat_exact(self):
        """compute_90 should cover sm_90."""
        arch_list = ["sm_50", "compute_90"]
        assert is_sm_in_arch_list("sm_90", arch_list) is True

    def test_ptx_no_forward_compat_older(self):
        """compute_90 should NOT cover sm_80 (older than compute target)."""
        arch_list = ["compute_90"]
        assert is_sm_in_arch_list("sm_80", arch_list) is False

    def test_empty_arch_list(self):
        assert is_sm_in_arch_list("sm_89", []) is False

    def test_invalid_sm_format(self):
        assert is_sm_in_arch_list("invalid", ["sm_50", "sm_60"]) is False

    def test_only_compute_entries(self):
        """Arch list with only compute entries should still work."""
        arch_list = ["compute_80"]
        assert is_sm_in_arch_list("sm_89", arch_list) is True
        assert is_sm_in_arch_list("sm_70", arch_list) is False
