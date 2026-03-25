"""
Cloud GPU instance recommender.

Suggests cloud GPU instances (AWS, GCP, Azure) based on VRAM requirements,
sorted by cost.
"""

import json
import os
from typing import Any, Dict, List


class CloudRecommender:
    """Recommend cloud GPU instances based on VRAM requirements."""

    def __init__(self):
        """Load cloud instance data from JSON."""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "cloud_instances.json",
        )
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.instances = data["instances"]

    def recommend(self, vram_mb: int) -> List[Dict[str, Any]]:
        """
        Return instances where total_vram_gb >= vram_mb/1024,
        sorted by approx_cost_hr ascending (cheapest first).

        Args:
            vram_mb: Required VRAM in megabytes

        Returns:
            List of instance dicts with provider, name, gpu_summary,
            total_vram_gb, approx_cost_hr, and headroom_gb.
        """
        required_gb = vram_mb / 1024
        results = []

        for inst in self.instances:
            if inst["total_vram_gb"] >= required_gb:
                gpu = inst["gpus"][0]
                if gpu["count"] == 1:
                    gpu_summary = f"1x {gpu['model']} ({gpu['vram_gb']}GB)"
                else:
                    gpu_summary = f"{gpu['count']}x {gpu['model']} ({gpu['vram_gb']}GB each)"

                results.append({
                    "provider": inst["provider"],
                    "name": inst["name"],
                    "gpu_summary": gpu_summary,
                    "total_vram_gb": inst["total_vram_gb"],
                    "approx_cost_hr": inst["approx_cost_hr"],
                    "headroom_gb": round(inst["total_vram_gb"] - required_gb, 1),
                })

        results.sort(key=lambda x: x["approx_cost_hr"])
        return results

    def recommend_for_model(self, vram_requirements: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Given vram_requirements dict (precision -> {vram_mb, ...}),
        return recommendations per quantization level.

        Args:
            vram_requirements: Dict mapping precision to requirement info
                (must contain 'vram_mb' key)

        Returns:
            Dict mapping precision to {vram_gb, instances} where instances
            is limited to top 5 cheapest. Only includes precisions where
            VRAM > 0.
        """
        result = {}

        for precision, req_info in vram_requirements.items():
            vram_mb = req_info.get("vram_mb", 0)
            if vram_mb <= 0:
                continue

            instances = self.recommend(vram_mb)
            result[precision] = {
                "vram_gb": round(vram_mb / 1024, 1),
                "instances": instances[:5],
            }

        return result
