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
from env_doctor.cli import check_compute_capability_compatibility


def _make_driver_result(gpu_cc="12.0", gpu_name="NVIDIA GeForce RTX 5090"):
    """Build a minimal mock driver DetectionResult."""
    class FakeResult:
        metadata = {
            "primary_gpu_compute_capability": gpu_cc,
            "primary_gpu_name": gpu_name,
        }
    return FakeResult()


def _make_torch_result(version="2.5.1", arch_list=None, cuda_version="12.1"):
    """Build a minimal mock torch DetectionResult."""
    if arch_list is None:
        arch_list = ["sm_50", "sm_60", "sm_70", "sm_80", "sm_86"]
    class FakeResult:
        pass
    r = FakeResult()
    r.version = version
    r.metadata = {"arch_list": arch_list, "cuda_version": cuda_version}
    return r


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


class TestCheckComputeCompatibilityMessages:
    """Tests for check_compute_capability_compatibility() message differentiation."""

    # --- Mismatch: hard failure (cuda_available=False) ---

    def test_hard_failure_message(self, capsys):
        driver = _make_driver_result(gpu_cc="12.0")
        torch = _make_torch_result(arch_list=["sm_50", "sm_60", "sm_70", "sm_80", "sm_86"])
        check_compute_capability_compatibility(driver, torch, cuda_available=False)
        out = capsys.readouterr().out
        assert "\u274c ARCHITECTURE MISMATCH" in out
        assert "ARCHITECTURE MISMATCH (Soft)" not in out
        assert "likely why torch.cuda.is_available() returns False" in out

    def test_soft_failure_message(self, capsys):
        driver = _make_driver_result(gpu_cc="7.5", gpu_name="NVIDIA GeForce RTX 2080")
        torch = _make_torch_result(
            version="1.13.1",
            arch_list=["sm_50", "sm_60", "sm_70"],
            cuda_version="11.7",
        )
        check_compute_capability_compatibility(driver, torch, cuda_available=True)
        out = capsys.readouterr().out
        assert "ARCHITECTURE MISMATCH (Soft)" in out
        assert "returned True via NVIDIA's driver-level PTX JIT" in out
        assert "\u274c ARCHITECTURE MISMATCH" not in out or "Soft" in out

    def test_none_available_treated_as_hard(self, capsys):
        driver = _make_driver_result(gpu_cc="12.0")
        torch = _make_torch_result(arch_list=["sm_50", "sm_60", "sm_70", "sm_80", "sm_86"])
        check_compute_capability_compatibility(driver, torch, cuda_available=None)
        out = capsys.readouterr().out
        assert "\u274c ARCHITECTURE MISMATCH" in out
        assert "ARCHITECTURE MISMATCH (Soft)" not in out
        assert "likely why torch.cuda.is_available() returns False" in out

    def test_compatible_unaffected(self, capsys):
        driver = _make_driver_result(gpu_cc="8.6", gpu_name="NVIDIA GeForce RTX 3090")
        torch = _make_torch_result(arch_list=["sm_50", "sm_60", "sm_70", "sm_80", "sm_86"])
        result = check_compute_capability_compatibility(driver, torch, cuda_available=True)
        out = capsys.readouterr().out
        assert "\u2705 COMPATIBLE" in out
        assert result["status"] == "compatible"

    # --- JSON / returned dict ---

    def test_json_hard_failure_contains_cuda_available_false(self):
        driver = _make_driver_result(gpu_cc="12.0")
        torch = _make_torch_result(arch_list=["sm_50", "sm_60", "sm_70", "sm_80", "sm_86"])
        result = check_compute_capability_compatibility(driver, torch, cuda_available=False)
        assert result["status"] == "mismatch"
        assert result["cuda_available"] is False

    def test_json_soft_failure_contains_cuda_available_true(self):
        driver = _make_driver_result(gpu_cc="7.5", gpu_name="NVIDIA GeForce RTX 2080")
        torch = _make_torch_result(
            version="1.13.1",
            arch_list=["sm_50", "sm_60", "sm_70"],
            cuda_version="11.7",
        )
        result = check_compute_capability_compatibility(driver, torch, cuda_available=True)
        assert result["status"] == "mismatch"
        assert result["cuda_available"] is True

    def test_json_compatible_contains_cuda_available(self):
        driver = _make_driver_result(gpu_cc="8.6", gpu_name="NVIDIA GeForce RTX 3090")
        torch = _make_torch_result(arch_list=["sm_50", "sm_60", "sm_70", "sm_80", "sm_86"])
        result = check_compute_capability_compatibility(driver, torch, cuda_available=True)
        assert result["status"] == "compatible"
        assert "cuda_available" in result
        assert result["cuda_available"] is True
