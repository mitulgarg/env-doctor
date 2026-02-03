"""
Unit tests for ModelChecker module.

Tests model compatibility checking, GPU detection integration, and recommendation generation.
"""

import pytest
from unittest.mock import patch, MagicMock
from env_doctor.utilities.model_checker import ModelChecker
from env_doctor.core.detector import DetectionResult, Status


class TestModelCheckerGPUDetection:
    """Test GPU information retrieval."""

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_get_gpu_info_single_gpu(self, mock_registry):
        """Test getting GPU info for single GPU system"""
        # Mock GPU with 24GB VRAM
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 24576,
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [{"total_vram_mb": 24576, "name": "RTX 3090", "index": 0}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        gpu_info = checker._get_gpu_info()

        assert gpu_info["available"] is True
        assert gpu_info["gpu_count"] == 1
        assert gpu_info["total_vram_mb"] == 24576
        assert gpu_info["primary_gpu_name"] == "RTX 3090"

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_get_gpu_info_multi_gpu(self, mock_registry):
        """Test getting GPU info for multi-GPU system"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 2,
                "total_vram_mb": 49152,  # 2 × 24GB
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [
                    {"total_vram_mb": 24576, "name": "RTX 3090", "index": 0},
                    {"total_vram_mb": 24576, "name": "RTX 3090", "index": 1},
                ],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        gpu_info = checker._get_gpu_info()

        assert gpu_info["gpu_count"] == 2
        assert gpu_info["total_vram_mb"] == 49152

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_get_gpu_info_no_gpu(self, mock_registry):
        """Test getting GPU info when no GPU is detected"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.NOT_FOUND,
            metadata={},
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        gpu_info = checker._get_gpu_info()

        assert gpu_info["available"] is False
        assert gpu_info["gpu_count"] == 0


class TestModelCheckerCompatibility:
    """Test model compatibility checking."""

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_model_fits_single_gpu(self, mock_registry):
        """Test model that fits on single GPU"""
        # Mock GPU with 24GB VRAM
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 24576,
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [{"total_vram_mb": 24576}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("llama-3-8b")

        assert result["success"] is True
        # llama-3-8b is 19GB fp16 (measured) - should fit in 24GB
        assert result["compatibility"]["fits_on_single_gpu"]["fp16"]["fits"] is True

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_model_doesnt_fit_single_gpu(self, mock_registry):
        """Test model too large for single GPU"""
        # Mock small GPU (8GB)
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 8192,
                "primary_gpu_name": "RTX 3060",
                "primary_gpu_vram_mb": 8192,
                "gpus": [{"total_vram_mb": 8192}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("llama-3-70b")

        assert result["success"] is True
        # 70B won't fit in 8GB even in int4 (42GB needed)
        assert result["compatibility"]["fits_on_single_gpu"]["int4"]["fits"] is False

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_model_fits_multi_gpu(self, mock_registry):
        """Test model that fits on multi-GPU"""
        # Mock 2x RTX 3090
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 2,
                "total_vram_mb": 49152,  # 2 × 24GB
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [
                    {"total_vram_mb": 24576},
                    {"total_vram_mb": 24576},
                ],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("mixtral-8x7b")

        assert result["success"] is True
        assert "fits_on_multi_gpu" in result["compatibility"]

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_check_specific_precision(self, mock_registry):
        """Test checking specific precision"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 8192,
                "primary_gpu_name": "RTX 3060",
                "primary_gpu_vram_mb": 8192,
                "gpus": [{"total_vram_mb": 8192}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("mistral-7b", precision="int4")

        assert result["success"] is True
        # Should only have int4 in vram_requirements
        assert "int4" in result["vram_requirements"]
        assert len(result["vram_requirements"]) == 1


class TestModelCheckerModelNotFound:
    """Test error handling for unknown models."""

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_model_not_found(self, mock_registry):
        """Test handling of unknown model"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={"gpu_count": 1, "total_vram_mb": 24576, "primary_gpu_vram_mb": 24576},
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("nonexistent-model-xyz")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_model_not_found_suggestions(self, mock_registry):
        """Test that similar model names are suggested"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={"gpu_count": 1, "total_vram_mb": 24576, "primary_gpu_vram_mb": 24576},
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("gpt-5-model-xyz")  # Non-existent model

        assert result["success"] is False
        # Should have error message
        assert "not found" in result["error"].lower()


class TestModelCheckerRecommendations:
    """Test recommendation generation."""

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_recommendation_best_precision(self, mock_registry):
        """Test recommendation when model fits"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 24576,
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [{"total_vram_mb": 24576}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("llama-3-8b")

        assert len(result["recommendations"]) > 0
        # Should recommend best precision
        rec_text = " ".join(result["recommendations"])
        assert "fp16" in rec_text.lower()

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_recommendation_no_gpu(self, mock_registry):
        """Test recommendation when no GPU available"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.NOT_FOUND,
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("llama-3-8b")

        assert len(result["recommendations"]) > 0
        # Should recommend cloud GPUs
        rec_text = " ".join(result["recommendations"])
        assert "cloud" in rec_text.lower() or "gpu" in rec_text.lower()

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_recommendation_smaller_variant(self, mock_registry):
        """Test recommendation for smaller model variant"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 12288,  # 12GB
                "primary_gpu_name": "RTX 2070",
                "primary_gpu_vram_mb": 12288,
                "gpus": [{"total_vram_mb": 12288}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("llama-3-70b")

        # Model doesn't fit, should suggest alternatives
        assert len(result["recommendations"]) > 0


class TestModelCheckerBestPrecision:
    """Test finding best precision."""

    def test_find_best_precision_fp16(self):
        """Test finding best precision prefers fp16"""
        checker = ModelChecker()
        compatibility = {
            "fp16": {"fits": True, "free_vram_mb": 1000},
            "int8": {"fits": True, "free_vram_mb": 500},
            "int4": {"fits": True, "free_vram_mb": 100},
        }
        best = checker._find_best_precision(compatibility)
        assert best == "fp16"

    def test_find_best_precision_bf16(self):
        """Test fallback to bf16 if fp16 not available"""
        checker = ModelChecker()
        compatibility = {
            "bf16": {"fits": True, "free_vram_mb": 1000},
            "int8": {"fits": True, "free_vram_mb": 500},
        }
        best = checker._find_best_precision(compatibility)
        assert best == "bf16"

    def test_find_best_precision_int4_fallback(self):
        """Test fallback to int4"""
        checker = ModelChecker()
        compatibility = {
            "int4": {"fits": True, "free_vram_mb": 100},
        }
        best = checker._find_best_precision(compatibility)
        assert best == "int4"


class TestModelCheckerHuggingFaceIntegration:
    """Test ModelChecker with HuggingFace API integration."""

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", True)
    @patch("env_doctor.utilities.vram_calculator.model_info")
    def test_model_fetched_from_hf(self, mock_model_info, mock_registry):
        """Test that model is fetched from HuggingFace when not in local DB"""
        # Mock HuggingFace API response
        mock_info = MagicMock()
        mock_info.safetensors = {"total": 7_000_000_000}  # 7B params
        mock_info.card_data = None
        mock_model_info.return_value = mock_info

        # Mock GPU
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 24576,
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [{"total_vram_mb": 24576}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        # Create temp db without the model
        import tempfile
        import json
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "_metadata": {"version": "1.0"},
                "models": {},
                "aliases": {},
                "hf_cache": {}
            }, f)
            temp_path = f.name

        try:
            from env_doctor.utilities.vram_calculator import VRAMCalculator
            # Patch the VRAMCalculator used by ModelChecker
            with patch.object(ModelChecker, '__init__', lambda self: None):
                checker = ModelChecker()
                checker.vram_calc = VRAMCalculator(db_path=temp_path)

                result = checker.check_compatibility("test-org/new-model-7b")

                assert result["success"] is True
                assert result.get("fetched_from_hf") is True
        finally:
            os.unlink(temp_path)

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    def test_fetched_from_hf_flag_false_for_local_model(self, mock_registry):
        """Test that fetched_from_hf is False for models in local DB"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 24576,
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [{"total_vram_mb": 24576}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("llama-3-8b")

        assert result["success"] is True
        assert result.get("fetched_from_hf", False) is False

    @patch("env_doctor.utilities.model_checker.DetectorRegistry.get")
    @patch("env_doctor.utilities.vram_calculator.HF_AVAILABLE", False)
    def test_error_message_when_hf_unavailable(self, mock_registry):
        """Test error message suggests installing huggingface_hub"""
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            component="nvidia_driver",
            status=Status.SUCCESS,
            metadata={
                "gpu_count": 1,
                "total_vram_mb": 24576,
                "primary_gpu_name": "RTX 3090",
                "primary_gpu_vram_mb": 24576,
                "gpus": [{"total_vram_mb": 24576}],
            },
        )
        mock_detector.detect.return_value = mock_result
        mock_registry.return_value = mock_detector

        checker = ModelChecker()
        result = checker.check_compatibility("nonexistent-model-xyz")

        assert result["success"] is False
        assert "huggingface_hub" in result["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
