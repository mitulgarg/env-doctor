"""
Model-hardware compatibility checker.

Determines if AI models can run on available GPU hardware and provides
compatibility analysis and recommendations.
"""

from typing import Dict, List, Optional, Any
from env_doctor.utilities.vram_calculator import VRAMCalculator
from env_doctor.core.registry import DetectorRegistry


class ModelChecker:
    """Check if AI models fit on available GPU hardware."""

    def __init__(self):
        """Initialize checker with VRAM calculator."""
        self.vram_calc = VRAMCalculator()

    def check_compatibility(
        self, model_name: str, precision: Optional[str] = None, _depth: int = 0
    ) -> Dict[str, Any]:
        """
        Check if model is compatible with available hardware.

        Args:
            model_name: Name of the model to check
            precision: Optional specific precision to check (fp32, fp16, bf16, int8, int4)
            _depth: Internal parameter to prevent infinite recursion when checking family variants

        Returns:
            Dict with keys:
                - success: bool - Whether model was found
                - error: str (if not success) - Error message
                - suggestions: list (if not success) - Similar model names
                - model_name: str - Normalized model name
                - model_info: dict - Model data from database
                - gpu_info: dict - Available GPU information
                - vram_requirements: dict - VRAM needed for each precision
                - compatibility: dict - Which precisions fit on GPU
                - recommendations: list - Actionable recommendations
        """
        # 1. Get GPU info from NvidiaDriverDetector
        gpu_info = self._get_gpu_info()

        # 2. Get model info from VRAMCalculator
        try:
            normalized_name = self.vram_calc._normalize_model_name(model_name)
            model_info = self.vram_calc.get_model_info(normalized_name)

            if not model_info:
                return {
                    "success": False,
                    "error": f"Model '{model_name}' not found in database",
                    "suggestions": self._suggest_similar_models(model_name),
                }
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "suggestions": self._suggest_similar_models(model_name),
            }

        # 3. Calculate VRAM for specific precision or all
        if precision:
            try:
                vram_reqs = {
                    precision: self.vram_calc.calculate_vram(normalized_name, precision)
                }
            except (ValueError, KeyError) as e:
                return {
                    "success": False,
                    "error": f"Error calculating VRAM for {precision}: {str(e)}",
                    "suggestions": [],
                }
        else:
            vram_reqs = self.vram_calc.calculate_all_precisions(normalized_name)

        # 4. Analyze compatibility
        compatibility = self._analyze_compatibility(vram_reqs, gpu_info)

        # 5. Generate recommendations (prevent recursion by passing depth)
        recommendations = self._generate_recommendations(
            normalized_name, model_info, vram_reqs, gpu_info, compatibility, _depth=_depth
        )

        return {
            "success": True,
            "model_name": normalized_name,
            "model_info": model_info,
            "gpu_info": gpu_info,
            "vram_requirements": vram_reqs,
            "compatibility": compatibility,
            "recommendations": recommendations,
        }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information from NvidiaDriverDetector.

        Returns:
            Dict with GPU information:
                - available: bool
                - gpu_count: int
                - total_vram_mb: int
                - primary_gpu_name: str or None
                - primary_gpu_vram_mb: int
                - gpus: list of GPU dicts
        """
        try:
            driver_detector = DetectorRegistry.get("nvidia_driver")
            result = driver_detector.detect()

            if not result.detected:
                return {
                    "available": False,
                    "gpu_count": 0,
                    "total_vram_mb": 0,
                    "primary_gpu_name": None,
                    "primary_gpu_vram_mb": 0,
                    "gpus": [],
                }

            metadata = result.metadata

            return {
                "available": True,
                "gpu_count": metadata.get("gpu_count", 0),
                "total_vram_mb": metadata.get("total_vram_mb", 0),
                "primary_gpu_name": metadata.get("primary_gpu_name"),
                "primary_gpu_vram_mb": metadata.get("primary_gpu_vram_mb", 0),
                "gpus": metadata.get("gpus", []),
            }
        except Exception:
            return {
                "available": False,
                "gpu_count": 0,
                "total_vram_mb": 0,
                "primary_gpu_name": None,
                "primary_gpu_vram_mb": 0,
                "gpus": [],
            }

    def _analyze_compatibility(
        self, vram_reqs: Dict[str, Dict], gpu_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze which precisions fit on available hardware.

        Args:
            vram_reqs: VRAM requirements for each precision
            gpu_info: Available GPU information

        Returns:
            Dict with:
                - fits_on_single_gpu: Which precisions fit on primary GPU
                - fits_on_multi_gpu: Which precisions fit on all GPUs (if multiple)
                - no_gpu_available: bool
        """
        if not gpu_info["available"]:
            return {
                "fits_on_single_gpu": {},
                "fits_on_multi_gpu": {},
                "no_gpu_available": True,
            }

        single_gpu_vram = gpu_info["primary_gpu_vram_mb"]
        total_vram = gpu_info["total_vram_mb"]

        fits_single = {}
        fits_multi = {}

        for precision, req_info in vram_reqs.items():
            required_mb = req_info["vram_mb"]

            # Single GPU check
            if required_mb <= single_gpu_vram:
                fits_single[precision] = {
                    "fits": True,
                    "free_vram_mb": single_gpu_vram - required_mb,
                }
            else:
                fits_single[precision] = {
                    "fits": False,
                    "shortage_mb": required_mb - single_gpu_vram,
                }

            # Multi-GPU check
            if gpu_info["gpu_count"] > 1:
                if required_mb <= total_vram:
                    fits_multi[precision] = {
                        "fits": True,
                        "free_vram_mb": total_vram - required_mb,
                        "gpus_needed": gpu_info["gpu_count"],
                    }
                else:
                    fits_multi[precision] = {
                        "fits": False,
                        "shortage_mb": required_mb - total_vram,
                    }

        return {
            "fits_on_single_gpu": fits_single,
            "fits_on_multi_gpu": fits_multi if gpu_info["gpu_count"] > 1 else {},
            "no_gpu_available": False,
        }

    def _generate_recommendations(
        self,
        model_name: str,
        model_info: Dict,
        vram_reqs: Dict,
        gpu_info: Dict,
        compatibility: Dict,
        _depth: int = 0,
    ) -> List[str]:
        """
        Generate actionable recommendations based on compatibility analysis.

        Args:
            model_name: Name of the model
            model_info: Model information from database
            vram_reqs: VRAM requirements
            gpu_info: Available GPU information
            compatibility: Compatibility analysis
            _depth: Internal parameter to prevent infinite recursion

        Returns:
            List of recommendation strings
        """
        recs = []

        # Check if any precision fits
        any_fits = any(
            p["fits"] for p in compatibility["fits_on_single_gpu"].values()
        )

        if not gpu_info["available"]:
            recs.append("No NVIDIA GPU detected. This model requires a GPU to run.")
            recs.append("Consider using cloud GPUs (AWS, GCP, RunPod, Lambda Labs)")
            return recs

        if any_fits:
            # Find best precision that fits (prefer fp16/bf16, then int8, then int4)
            best_precision = self._find_best_precision(
                compatibility["fits_on_single_gpu"]
            )
            recs.append(
                f"Use {best_precision.upper()} for best quality on your GPU"
            )
            return recs

        # Model doesn't fit - suggest alternatives
        # Only check family variants at depth 0 to prevent infinite recursion
        if _depth == 0:
            # 1. Try to find smaller variants from same family
            family_variants = self.vram_calc.get_model_family_variants(model_name)

            for variant in family_variants:
                if variant == model_name:
                    continue

                variant_check = self.check_compatibility(variant, _depth=_depth + 1)
                if variant_check.get("success"):
                    variant_compat = variant_check["compatibility"]
                    if any(
                        p["fits"] for p in variant_compat["fits_on_single_gpu"].values()
                    ):
                        variant_params = variant_check["model_info"]["params_b"]
                        recs.append(
                            f"Try smaller variant: {variant} ({variant_params}B params)"
                        )
                        break

        # 2. Multi-GPU suggestion
        if gpu_info["gpu_count"] > 1:
            multi_fits = any(
                p["fits"] for p in compatibility["fits_on_multi_gpu"].values()
            )
            if multi_fits:
                best_multi = self._find_best_precision(
                    compatibility["fits_on_multi_gpu"]
                )
                recs.append(
                    f"Use model parallelism across {gpu_info['gpu_count']} GPUs "
                    f"in {best_multi.upper()} (requires accelerate or DeepSpeed)"
                )
            else:
                # Not enough VRAM even on multi-GPU
                min_req = min(r["vram_mb"] for r in vram_reqs.values())
                gpus_needed = (min_req // gpu_info["primary_gpu_vram_mb"]) + 1
                recs.append(
                    f"Need {gpus_needed}x {gpu_info['primary_gpu_name']} "
                    f"({min_req // 1024}GB total)"
                )
        else:
            # Single GPU - calculate upgrade requirement
            min_req = min(r["vram_mb"] for r in vram_reqs.values())
            min_req_gb = min_req // 1024

            current_vram = gpu_info["primary_gpu_vram_mb"] // 1024
            recs.append(
                f"Upgrade GPU: current {current_vram}GB, need {min_req_gb}GB minimum"
            )

        # 3. Cloud GPU suggestion
        recs.append("Consider cloud GPUs: A100 (80GB), H100 (80GB), or RTX 6000 Ada")

        return recs

    def _find_best_precision(self, compatibility: Dict[str, Dict]) -> str:
        """
        Find highest quality precision that fits.

        Preference order: fp16 > bf16 > int8 > int4 > fp32

        Args:
            compatibility: Compatibility info for each precision

        Returns:
            Best fitting precision name
        """
        preference = ["fp16", "bf16", "int8", "int4", "fp32"]

        for prec in preference:
            if prec in compatibility and compatibility[prec].get("fits"):
                return prec

        return "int4"  # Fallback to smallest

    def _suggest_similar_models(self, query: str) -> List[str]:
        """
        Suggest models with similar names (fuzzy matching).

        Args:
            query: Search query

        Returns:
            List of similar model names
        """
        suggestions = []
        query_lower = query.lower()

        for name in self.vram_calc.db["models"].keys():
            if query_lower in name or name in query_lower:
                suggestions.append(name)

        # Also check aliases
        for alias, target in self.vram_calc.db.get("aliases", {}).items():
            if query_lower in alias or alias in query_lower:
                suggestions.append(f"{alias} (alias for {target})")

        return suggestions[:5]  # Top 5 matches
