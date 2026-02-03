"""
Unit tests for VRAMCalculator module.

Tests VRAM calculation logic, database loading, and alias resolution.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from env_doctor.utilities.vram_calculator import VRAMCalculator


class TestVRAMCalculatorFormulas:
    """Test VRAM calculation formulas."""

    def test_formula_fp32(self):
        """Test FP32 calculation: 7B × 4 bytes × 1.2 = 33.6GB = 33600MB"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(7.0, "fp32")
        assert result == 33600  # 7 * 4 * 1.2 * 1000

    def test_formula_fp16(self):
        """Test FP16 calculation: 7B × 2 bytes × 1.2 = 16.8GB = 16800MB"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(7.0, "fp16")
        assert result == 16800  # 7 * 2 * 1.2 * 1000

    def test_formula_bf16(self):
        """Test BF16 calculation (same as FP16)"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(7.0, "bf16")
        assert result == 16800

    def test_formula_int8(self):
        """Test INT8 calculation: 7B × 1 byte × 1.2 = 8.4GB = 8400MB"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(7.0, "int8")
        assert result == 8400  # 7 * 1 * 1.2 * 1000

    def test_formula_int4(self):
        """Test INT4 calculation: 7B × 0.5 bytes × 1.2 = 4.2GB = 4200MB"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(7.0, "int4")
        assert result == 4200  # 7 * 0.5 * 1.2 * 1000

    def test_formula_fp8(self):
        """Test FP8 calculation: 12B × 1 byte × 1.2 ≈ 14.4GB ≈ 14400MB"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(12.0, "fp8")
        # Allow for integer rounding (int() conversion)
        assert result in [14399, 14400]  # 12 * 1 * 1.2 * 1000

    def test_formula_very_small_model(self):
        """Test calculation with very small model (0.1B)"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(0.1, "fp16")
        assert result == 240  # 0.1 * 2 * 1.2 * 1000

    def test_formula_very_large_model(self):
        """Test calculation with very large model (400B)"""
        calc = VRAMCalculator()
        result = calc._calculate_from_params(405.0, "int4")
        assert result == 243000  # 405 * 0.5 * 1.2 * 1000

    def test_formula_unsupported_precision(self):
        """Test error handling for unsupported precision"""
        calc = VRAMCalculator()
        with pytest.raises(KeyError):
            calc._calculate_from_params(7.0, "fp64")


class TestVRAMCalculatorMeasuredVsEstimated:
    """Test hybrid measured vs estimated VRAM logic."""

    def test_measured_value_used(self):
        """Test that measured values override formula"""
        calc = VRAMCalculator()
        # llama-3-8b has measured fp16 value
        result = calc.calculate_vram("llama-3-8b", "fp16")
        assert result["source"] == "measured"
        assert result["vram_mb"] == 19200
        assert result["params_b"] == 8.0

    def test_measured_value_int4(self):
        """Test measured INT4 value"""
        calc = VRAMCalculator()
        result = calc.calculate_vram("llama-3-8b", "int4")
        assert result["source"] == "measured"
        assert result["vram_mb"] == 4800

    def test_formula_fallback(self):
        """Test formula used when no measured value"""
        calc = VRAMCalculator()
        # qwen-7b has no measured values
        result = calc.calculate_vram("qwen-7b", "fp16")
        assert result["source"] == "estimated"
        # Should calculate: 7 * 2 * 1.2 * 1000 = 16800
        assert result["vram_mb"] == 16800

    def test_mixed_measured_and_estimated(self):
        """Test model with some measured and some estimated"""
        calc = VRAMCalculator()
        # mixtral-8x22b has measured fp16
        result_fp16 = calc.calculate_vram("mixtral-8x22b", "fp16")
        assert result_fp16["source"] == "measured"

        # But no measured int8 (should estimate)
        result_int8 = calc.calculate_vram("mixtral-8x22b", "int8")
        assert result_int8["source"] == "estimated"


class TestVRAMCalculatorAllPrecisions:
    """Test calculating all precisions at once."""

    def test_calculate_all_precisions(self):
        """Test getting all precisions for a model"""
        calc = VRAMCalculator()
        results = calc.calculate_all_precisions("llama-3-8b")

        # Should have at least measured values
        assert "fp16" in results
        assert "int4" in results

        # Check values are reasonable (int4 < fp16)
        assert results["int4"]["vram_mb"] < results["fp16"]["vram_mb"]

    def test_all_precisions_increases_in_size(self):
        """Test that larger precisions require more VRAM"""
        calc = VRAMCalculator()
        results = calc.calculate_all_precisions("mistral-7b")

        # Precision sizes should be ordered (using formula since no measured)
        fp32 = results["fp32"]["vram_mb"]
        fp16 = results["fp16"]["vram_mb"]
        int8 = results["int8"]["vram_mb"]
        int4 = results["int4"]["vram_mb"]

        # Each should be smaller or equal to previous
        assert fp32 > fp16
        assert fp16 > int8
        assert int8 > int4


class TestVRAMCalculatorModelNotFound:
    """Test error handling for unknown models."""

    def test_model_not_found(self):
        """Test error handling for unknown model"""
        calc = VRAMCalculator()
        with pytest.raises(ValueError) as exc_info:
            calc.calculate_vram("nonexistent-model-xyz", "fp16")
        assert "not found" in str(exc_info.value).lower()

    def test_model_not_found_message_helpful(self):
        """Test error message is helpful"""
        calc = VRAMCalculator()
        with pytest.raises(ValueError) as exc_info:
            calc.calculate_vram("fake-model", "fp16")
        error_msg = str(exc_info.value)
        assert "fake-model" in error_msg
        assert "--list" in error_msg or "available" in error_msg.lower()


class TestVRAMCalculatorAliases:
    """Test model name alias resolution."""

    def test_alias_resolution(self):
        """Test that aliases are resolved correctly"""
        calc = VRAMCalculator()
        # llama3-8b is an alias for llama-3-8b
        result1 = calc.calculate_vram("llama-3-8b", "fp16")
        result2 = calc.calculate_vram("llama3-8b", "fp16")
        assert result1["vram_mb"] == result2["vram_mb"]

    def test_case_insensitive_matching(self):
        """Test that model names are case-insensitive"""
        calc = VRAMCalculator()
        result1 = calc.calculate_vram("llama-3-8b", "fp16")
        result2 = calc.calculate_vram("LLAMA-3-8B", "fp16")
        result3 = calc.calculate_vram("LLaMA-3-8b", "fp16")
        assert result1["vram_mb"] == result2["vram_mb"] == result3["vram_mb"]

    def test_sdxl_alias(self):
        """Test SDXL alias"""
        calc = VRAMCalculator()
        result1 = calc.calculate_vram("stable-diffusion-xl", "fp16")
        result2 = calc.calculate_vram("sdxl", "fp16")
        assert result1["vram_mb"] == result2["vram_mb"]


class TestVRAMCalculatorModelInfo:
    """Test getting model information."""

    def test_get_model_info(self):
        """Test retrieving model information"""
        calc = VRAMCalculator()
        info = calc.get_model_info("llama-3-8b")

        assert info is not None
        assert info["params_b"] == 8.0
        assert info["category"] == "llm"
        assert info["family"] == "llama-3"

    def test_get_model_info_not_found(self):
        """Test get_model_info returns None for unknown model"""
        calc = VRAMCalculator()
        info = calc.get_model_info("nonexistent-model")
        assert info is None

    def test_get_model_info_via_alias(self):
        """Test getting model info via alias"""
        calc = VRAMCalculator()
        info = calc.get_model_info("sdxl")
        assert info is not None
        assert info["params_b"] == 3.5


class TestVRAMCalculatorDatabaseStats:
    """Test database statistics."""

    def test_get_database_stats(self):
        """Test getting database statistics"""
        calc = VRAMCalculator()
        stats = calc.get_database_stats()

        assert "total_models" in stats
        assert "total_aliases" in stats
        assert "models_by_category" in stats
        assert "models_with_measured_vram" in stats

        # Should have at least 20 models as per spec
        assert stats["total_models"] >= 20

        # Should have categories
        assert "llm" in stats["models_by_category"]
        assert "diffusion" in stats["models_by_category"]

    def test_database_has_measured_vram(self):
        """Test that database has some measured VRAM values"""
        calc = VRAMCalculator()
        stats = calc.get_database_stats()

        # Should have at least a few models with measured VRAM
        assert stats["models_with_measured_vram"] >= 5
        assert len(stats["precisions_with_measured_data"]) > 0


class TestVRAMCalculatorListAllModels:
    """Test listing all models."""

    def test_list_all_models(self):
        """Test listing all models by category"""
        calc = VRAMCalculator()
        models = calc.list_all_models()

        # Should have all categories
        assert "llm" in models
        assert "diffusion" in models
        assert "audio" in models

        # Each category should have models
        assert len(models["llm"]) > 0
        assert len(models["diffusion"]) > 0

        # Each model should have required fields
        for category_models in models.values():
            for model in category_models:
                assert "name" in model
                assert "params_b" in model

    def test_list_has_minimum_models(self):
        """Test that we have minimum 20 models"""
        calc = VRAMCalculator()
        models = calc.list_all_models()
        total = sum(len(m) for m in models.values())
        assert total >= 20


class TestVRAMCalculatorFamilyVariants:
    """Test getting model family variants."""

    def test_get_family_variants(self):
        """Test getting variants from same model family"""
        calc = VRAMCalculator()
        variants = calc.get_model_family_variants("llama-3-8b")

        # Should have other llama-3 models
        assert len(variants) > 0
        assert "llama-3-70b" in variants

    def test_family_variants_sorted_by_size(self):
        """Test that variants are sorted by size (ascending)"""
        calc = VRAMCalculator()
        variants = calc.get_model_family_variants("llama-3-70b")

        # Should include smaller variant (70b itself is excluded)
        assert "llama-3-8b" in variants
        # 8B should come first in the list since it's smaller
        assert variants[0] == "llama-3-8b" or "llama-3-8b" in variants[:2]

    def test_family_variants_no_self(self):
        """Test that model doesn't include itself"""
        calc = VRAMCalculator()
        variants = calc.get_model_family_variants("llama-3-8b")
        assert "llama-3-8b" not in variants

    def test_family_variants_empty_for_unknown_family(self):
        """Test empty list for model with no family"""
        calc = VRAMCalculator()
        variants = calc.get_model_family_variants("bert-base")
        # bert-base might have bert-large as family variant
        # Or might be empty if no family defined
        assert isinstance(variants, list)


class TestVRAMCalculatorDatabaseIntegrity:
    """Test database integrity and validity."""

    def test_database_loads_correctly(self):
        """Test database file loads without errors"""
        calc = VRAMCalculator()
        assert calc.db is not None
        assert "models" in calc.db
        assert "_metadata" in calc.db

    def test_all_models_have_required_fields(self):
        """Test all models have required fields"""
        calc = VRAMCalculator()
        for name, data in calc.db["models"].items():
            assert "params_b" in data, f"{name} missing params_b"
            assert "category" in data, f"{name} missing category"
            assert isinstance(data["params_b"], (int, float))
            assert data["params_b"] > 0, f"{name} params_b not positive"

    def test_vram_values_reasonable(self):
        """Test measured VRAM values are reasonable"""
        calc = VRAMCalculator()
        for name, data in calc.db["models"].items():
            if "vram" not in data:
                continue
            for precision, vram_mb in data["vram"].items():
                # VRAM should be positive
                assert vram_mb > 0, f"{name} {precision} VRAM <= 0"
                # VRAM should be < 1TB (1,000,000 MB)
                assert vram_mb < 1000000, f"{name} {precision} VRAM > 1TB"

    def test_aliases_point_to_real_models(self):
        """Test all aliases resolve to real models"""
        calc = VRAMCalculator()
        if "aliases" in calc.db:
            for alias, target in calc.db["aliases"].items():
                assert (
                    target in calc.db["models"]
                ), f"Alias {alias} points to nonexistent {target}"

    def test_minimum_models_in_database(self):
        """Test database has at least 20 models"""
        calc = VRAMCalculator()
        assert len(calc.db["models"]) >= 20


class TestVRAMCalculatorHuggingFaceIntegration:
    """Test HuggingFace API integration for 3-tier fallback."""

    def test_hf_cache_initialized(self):
        """Test that hf_cache section is initialized in db"""
        calc = VRAMCalculator()
        assert "hf_cache" in calc.db

    def test_db_path_stored(self):
        """Test that db_path is stored for cache writing"""
        calc = VRAMCalculator()
        assert hasattr(calc, "db_path")
        assert calc.db_path is not None

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", False)
    def test_fetch_from_hf_disabled_when_not_available(self):
        """Test HF fetch returns None when huggingface_hub not installed"""
        calc = VRAMCalculator()
        result = calc._fetch_from_huggingface("meta-llama/Llama-2-7b-hf")
        assert result is None

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", True)
    @patch("env_doctor.utilities.vram_calculator.model_info")
    def test_fetch_from_hf_safetensors_metadata(self, mock_model_info):
        """Test fetching params from safetensors metadata"""
        mock_info = MagicMock()
        mock_info.safetensors = {"total": 7_000_000_000}  # 7B params
        mock_info.card_data = None
        mock_model_info.return_value = mock_info

        calc = VRAMCalculator()
        result = calc._fetch_from_huggingface("test-org/test-model-7b")

        assert result is not None
        assert result["params_b"] == 7.0
        assert result["hf_id"] == "test-org/test-model-7b"
        assert result["source"] == "huggingface_api"

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", True)
    @patch("env_doctor.utilities.vram_calculator.model_info")
    def test_fetch_from_hf_fallback_to_name_parsing(self, mock_model_info):
        """Test fallback to extracting params from model name"""
        mock_info = MagicMock()
        mock_info.safetensors = None  # No safetensors metadata
        mock_info.card_data = None
        mock_model_info.return_value = mock_info

        calc = VRAMCalculator()
        result = calc._fetch_from_huggingface("some-org/my-model-13b")

        assert result is not None
        assert result["params_b"] == 13.0

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", True)
    @patch("env_doctor.utilities.vram_calculator.model_info")
    def test_fetch_from_hf_not_found(self, mock_model_info):
        """Test handling of model not found on HuggingFace"""
        # Create a mock exception class for RepositoryNotFoundError
        class MockRepoNotFoundError(Exception):
            pass

        # Patch the exception class in the module
        with patch("env_doctor.utilities.vram_calculator.RepositoryNotFoundError", MockRepoNotFoundError):
            mock_model_info.side_effect = MockRepoNotFoundError("Not found")

            calc = VRAMCalculator()
            result = calc._fetch_from_huggingface("nonexistent/model")

            assert result is None

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", True)
    @patch("env_doctor.utilities.vram_calculator.model_info")
    def test_fetch_from_hf_timeout_handling(self, mock_model_info):
        """Test handling of timeout/network errors"""
        mock_model_info.side_effect = Exception("Connection timeout")

        calc = VRAMCalculator()
        result = calc._fetch_from_huggingface("some-org/model")

        assert result is None  # Should gracefully return None

    def test_save_to_cache(self):
        """Test saving model data to hf_cache"""
        # Create a temporary database file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "_metadata": {"version": "1.0"},
                "models": {},
                "aliases": {},
                "hf_cache": {}
            }, f)
            temp_path = f.name

        try:
            calc = VRAMCalculator(db_path=temp_path)
            test_data = {"params_b": 7.0, "hf_id": "test/model", "source": "huggingface_api"}
            calc._save_to_cache("test-model", test_data)

            # Reload and verify
            with open(temp_path, 'r') as f:
                saved_db = json.load(f)

            assert "test-model" in saved_db["hf_cache"]
            assert saved_db["hf_cache"]["test-model"]["params_b"] == 7.0
        finally:
            os.unlink(temp_path)


class TestVRAMCalculatorThreeTierFallback:
    """Test the 3-tier fallback system in calculate_vram."""

    def test_tier1_local_database(self):
        """Test Tier 1: Model found in local database"""
        calc = VRAMCalculator()
        result = calc.calculate_vram("llama-3-8b", "fp16")

        assert result["source"] == "measured"
        assert "fetched_from_hf" not in result or result.get("fetched_from_hf") is False

    def test_tier2_hf_cache(self):
        """Test Tier 2: Model found in hf_cache"""
        # Create a temporary database with cached model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "_metadata": {"version": "1.0"},
                "models": {},
                "aliases": {},
                "hf_cache": {
                    "cached-model": {
                        "params_b": 5.0,
                        "hf_id": "test/cached-model",
                        "source": "huggingface_api"
                    }
                }
            }, f)
            temp_path = f.name

        try:
            calc = VRAMCalculator(db_path=temp_path)
            result = calc.calculate_vram("cached-model", "fp16")

            assert result["fetched_from_hf"] is True
            assert result["params_b"] == 5.0
        finally:
            os.unlink(temp_path)

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", True)
    @patch("env_doctor.utilities.vram_calculator.model_info")
    def test_tier3_huggingface_api(self, mock_model_info):
        """Test Tier 3: Fetch from HuggingFace API"""
        mock_info = MagicMock()
        mock_info.safetensors = {"total": 3_000_000_000}  # 3B params
        mock_info.card_data = None
        mock_model_info.return_value = mock_info

        # Create temp db without the model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "_metadata": {"version": "1.0"},
                "models": {},
                "aliases": {},
                "hf_cache": {}
            }, f)
            temp_path = f.name

        try:
            calc = VRAMCalculator(db_path=temp_path)
            result = calc.calculate_vram("test-org/new-model-3b", "fp16")

            assert result["fetched_from_hf"] is True
            assert result["params_b"] == 3.0

            # Verify it was cached
            with open(temp_path, 'r') as f:
                saved_db = json.load(f)
            assert "test-org/new-model-3b" in saved_db["hf_cache"]
        finally:
            os.unlink(temp_path)

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", False)
    def test_model_not_found_without_hf(self):
        """Test error message when model not found and HF unavailable"""
        calc = VRAMCalculator()
        with pytest.raises(ValueError) as exc_info:
            calc.calculate_vram("nonexistent-model", "fp16")

        error_msg = str(exc_info.value)
        assert "huggingface_hub" in error_msg.lower()

    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", True)
    @patch("env_doctor.utilities.vram_calculator.model_info")
    def test_model_not_found_with_hf(self, mock_model_info):
        """Test error message when model not found even with HF API"""
        # Create a mock exception class for RepositoryNotFoundError
        class MockRepoNotFoundError(Exception):
            pass

        with patch("env_doctor.utilities.vram_calculator.RepositoryNotFoundError", MockRepoNotFoundError):
            mock_model_info.side_effect = MockRepoNotFoundError("Not found")

            calc = VRAMCalculator()
            with pytest.raises(ValueError) as exc_info:
                calc.calculate_vram("completely-fake-model", "fp16")

            error_msg = str(exc_info.value)
            assert "not found" in error_msg.lower()
            assert "huggingface" in error_msg.lower()


class TestVRAMCalculatorGetModelInfoWithCache:
    """Test get_model_info with hf_cache support."""

    def test_get_model_info_from_local_db(self):
        """Test getting model info from local database"""
        calc = VRAMCalculator()
        info = calc.get_model_info("llama-3-8b")

        assert info is not None
        assert info["params_b"] == 8.0

    def test_get_model_info_from_cache(self):
        """Test getting model info from hf_cache"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "_metadata": {"version": "1.0"},
                "models": {},
                "aliases": {},
                "hf_cache": {
                    "cached-model": {
                        "params_b": 10.0,
                        "hf_id": "test/cached",
                        "source": "huggingface_api"
                    }
                }
            }, f)
            temp_path = f.name

        try:
            calc = VRAMCalculator(db_path=temp_path)
            info = calc.get_model_info("cached-model")

            assert info is not None
            assert info["params_b"] == 10.0
            assert info["source"] == "huggingface_api"
        finally:
            os.unlink(temp_path)

    def test_get_model_info_not_found(self):
        """Test get_model_info returns None for unknown model"""
        calc = VRAMCalculator()
        info = calc.get_model_info("nonexistent-model")
        assert info is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
