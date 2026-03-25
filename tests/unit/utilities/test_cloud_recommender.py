"""Tests for CloudRecommender utility."""

import pytest
from env_doctor.utilities.cloud_recommender import CloudRecommender


@pytest.fixture
def recommender():
    """Create a CloudRecommender instance."""
    return CloudRecommender()


class TestCloudRecommenderInit:
    """Tests for CloudRecommender initialization."""

    def test_loads_instances(self, recommender):
        """Should load cloud instances from JSON."""
        assert len(recommender.instances) > 0

    def test_instances_have_required_fields(self, recommender):
        """Each instance should have provider, name, gpus, total_vram_gb, approx_cost_hr."""
        for inst in recommender.instances:
            assert "provider" in inst
            assert "name" in inst
            assert "gpus" in inst
            assert "total_vram_gb" in inst
            assert "approx_cost_hr" in inst


class TestRecommend:
    """Tests for recommend() method."""

    def test_returns_sorted_by_cost(self, recommender):
        """Results should be sorted by approx_cost_hr ascending."""
        results = recommender.recommend(16 * 1024)  # 16GB
        costs = [r["approx_cost_hr"] for r in results]
        assert costs == sorted(costs)

    def test_filters_insufficient_vram(self, recommender):
        """Should only return instances with enough total VRAM."""
        vram_mb = 80 * 1024  # 80GB
        results = recommender.recommend(vram_mb)
        for r in results:
            assert r["total_vram_gb"] >= 80

    def test_small_vram_returns_many(self, recommender):
        """Small VRAM requirement should return many instances."""
        results = recommender.recommend(1 * 1024)  # 1GB
        assert len(results) > 5

    def test_very_high_vram_returns_empty(self, recommender):
        """VRAM beyond any instance should return empty list."""
        results = recommender.recommend(10_000 * 1024)  # 10TB
        assert results == []

    def test_headroom_calculation(self, recommender):
        """Headroom should be total_vram_gb minus required GB."""
        vram_mb = 20 * 1024  # 20GB
        results = recommender.recommend(vram_mb)
        for r in results:
            expected_headroom = round(r["total_vram_gb"] - 20.0, 1)
            assert r["headroom_gb"] == expected_headroom

    def test_result_has_required_keys(self, recommender):
        """Each result should have all expected keys."""
        results = recommender.recommend(16 * 1024)
        assert len(results) > 0
        for r in results:
            assert "provider" in r
            assert "name" in r
            assert "gpu_summary" in r
            assert "total_vram_gb" in r
            assert "approx_cost_hr" in r
            assert "headroom_gb" in r

    def test_gpu_summary_format_single(self, recommender):
        """Single-GPU instances should show '1x MODEL (XGB)'."""
        results = recommender.recommend(1 * 1024)
        single_gpu = [r for r in results if r["gpu_summary"].startswith("1x")]
        assert len(single_gpu) > 0
        for r in single_gpu:
            assert "1x " in r["gpu_summary"]
            assert "GB)" in r["gpu_summary"]

    def test_gpu_summary_format_multi(self, recommender):
        """Multi-GPU instances should show 'Nx MODEL (XGB each)'."""
        results = recommender.recommend(100 * 1024)  # need multi-GPU
        multi_gpu = [r for r in results if not r["gpu_summary"].startswith("1x")]
        assert len(multi_gpu) > 0
        for r in multi_gpu:
            assert "each)" in r["gpu_summary"]


class TestRecommendForModel:
    """Tests for recommend_for_model() method."""

    def test_returns_per_precision(self, recommender):
        """Should return recommendations for each precision."""
        vram_reqs = {
            "fp16": {"vram_mb": 19200},
            "int8": {"vram_mb": 9600},
            "int4": {"vram_mb": 4800},
        }
        result = recommender.recommend_for_model(vram_reqs)
        assert "fp16" in result
        assert "int8" in result
        assert "int4" in result

    def test_each_precision_has_vram_gb_and_instances(self, recommender):
        """Each precision entry should have vram_gb and instances."""
        vram_reqs = {"fp16": {"vram_mb": 19200}}
        result = recommender.recommend_for_model(vram_reqs)
        assert "vram_gb" in result["fp16"]
        assert "instances" in result["fp16"]
        assert result["fp16"]["vram_gb"] == round(19200 / 1024, 1)

    def test_limits_to_top_5(self, recommender):
        """Should return at most 5 instances per precision."""
        vram_reqs = {"int4": {"vram_mb": 1024}}  # 1GB - many match
        result = recommender.recommend_for_model(vram_reqs)
        assert len(result["int4"]["instances"]) <= 5

    def test_skips_zero_vram(self, recommender):
        """Should skip precisions with vram_mb <= 0."""
        vram_reqs = {
            "fp16": {"vram_mb": 19200},
            "broken": {"vram_mb": 0},
        }
        result = recommender.recommend_for_model(vram_reqs)
        assert "fp16" in result
        assert "broken" not in result

    def test_instances_sorted_by_cost(self, recommender):
        """Instances within each precision should be sorted by cost."""
        vram_reqs = {"fp16": {"vram_mb": 19200}}
        result = recommender.recommend_for_model(vram_reqs)
        costs = [i["approx_cost_hr"] for i in result["fp16"]["instances"]]
        assert costs == sorted(costs)
