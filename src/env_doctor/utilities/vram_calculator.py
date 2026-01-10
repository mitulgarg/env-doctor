"""
VRAM calculation utilities for AI models.

Provides hybrid approach:
1. Measured VRAM values (when available in database) - most accurate
2. Formula-based estimation (parameter-based calculation) - good enough for any model
"""

import json
import os
from typing import Dict, List, Optional, Any


class VRAMCalculator:
    """Calculate VRAM requirements for AI models."""

    # Overhead multiplier for activations, KV cache, framework overhead, etc.
    OVERHEAD_MULTIPLIER = 1.2

    # Bytes per parameter by precision
    BYTES_PER_PARAM = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
        "fp8": 1.0,
    }

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize calculator with model database.

        Args:
            db_path: Path to model_requirements.json. If None, uses default bundled path.
        """
        if db_path is None:
            # Load from package data using relative path
            # Go up two levels from utilities/ to env_doctor/
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_path, "data", "model_requirements.json")

        try:
            with open(db_path, "r") as f:
                self.db = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model database not found at {db_path}. "
                "Please ensure model_requirements.json is installed correctly."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model database: {e}")

    def calculate_vram(
        self, model_name: str, precision: str = "fp16"
    ) -> Dict[str, Any]:
        """
        Calculate VRAM requirement for a model.

        Strategy:
        1. Normalize model name (lowercase, resolve aliases)
        2. Check if model exists in database
        3. If measured VRAM exists for precision: use it (most accurate)
        4. Otherwise: calculate using formula (good enough)

        Args:
            model_name: Name of the model (e.g., "llama-3-8b")
            precision: Precision level (fp32, fp16, bf16, int8, int4, fp8)

        Returns:
            Dict with keys:
                - vram_mb: Required VRAM in MB
                - source: "measured" or "estimated"
                - params_b: Model parameter count in billions
                - formula (if estimated): The calculation formula used

        Raises:
            ValueError: If model not found in database
            KeyError: If precision not supported
        """
        # Normalize model name (lowercase, resolve aliases)
        normalized_name = self._normalize_model_name(model_name)

        # Check if model exists
        if normalized_name not in self.db["models"]:
            raise ValueError(
                f"Model '{model_name}' not found in database. "
                f"Use 'env-doctor model --list' to see available models."
            )

        model_data = self.db["models"][normalized_name]
        params_b = model_data["params_b"]

        # 1. Try measured VRAM first (highest accuracy)
        if "vram" in model_data and precision in model_data["vram"]:
            return {
                "vram_mb": model_data["vram"][precision],
                "source": "measured",
                "params_b": params_b,
            }

        # 2. Fallback to formula-based calculation
        vram_mb = self._calculate_from_params(params_b, precision)

        return {
            "vram_mb": vram_mb,
            "source": "estimated",
            "params_b": params_b,
            "formula": f"{params_b}B × {self.BYTES_PER_PARAM[precision]} bytes/param × {self.OVERHEAD_MULTIPLIER} overhead",
        }

    def calculate_all_precisions(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Calculate VRAM for all supported precisions.

        Args:
            model_name: Name of the model

        Returns:
            Dict mapping precision -> vram_info
            Example: {"fp16": {...}, "int8": {...}, "int4": {...}}
        """
        results = {}

        for precision in ["fp32", "fp16", "bf16", "int8", "int4", "fp8"]:
            try:
                results[precision] = self.calculate_vram(model_name, precision)
            except (ValueError, KeyError):
                # Skip precisions that aren't supported
                pass

        return results

    def _calculate_from_params(self, params_b: float, precision: str) -> int:
        """
        Formula-based VRAM estimation.

        Formula: params_billions × bytes_per_param × overhead × 1000 (GB→MB)

        Overhead (1.2x) accounts for:
        - Activation memory during forward pass
        - KV cache for transformers
        - Framework overhead (PyTorch, TensorFlow, etc.)
        - Gradient buffers if fine-tuning

        Args:
            params_b: Model size in billions of parameters
            precision: Precision level

        Returns:
            Estimated VRAM in MB
        """
        if precision not in self.BYTES_PER_PARAM:
            raise KeyError(
                f"Unsupported precision: {precision}. "
                f"Supported: {', '.join(self.BYTES_PER_PARAM.keys())}"
            )

        bytes_per_param = self.BYTES_PER_PARAM[precision]
        vram_gb = params_b * bytes_per_param * self.OVERHEAD_MULTIPLIER
        vram_mb = int(vram_gb * 1000)  # Convert GB to MB

        return vram_mb

    def _normalize_model_name(self, name: str) -> str:
        """
        Normalize model name and resolve aliases.

        Examples:
            "LLaMA-3-8B" → "llama-3-8b"
            "llama3-8b" → "llama-3-8b" (via alias)
            "SDXL" → "stable-diffusion-xl" (via alias)

        Args:
            name: Raw model name

        Returns:
            Normalized model name
        """
        name = name.lower().strip()

        # Check aliases
        aliases = self.db.get("aliases", {})
        if name in aliases:
            return aliases[name]

        return name

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full model information from database.

        Args:
            model_name: Name of the model

        Returns:
            Model data dict, or None if not found
        """
        normalized_name = self._normalize_model_name(model_name)
        return self.db["models"].get(normalized_name)

    def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all models grouped by category.

        Returns:
            Dict mapping category -> list of models
            Example:
            {
                "llm": [{"name": "llama-3-8b", "params_b": 8.0, "hf_id": "..."}, ...],
                "diffusion": [...]
            }
        """
        models_by_category = {}

        for name, data in self.db["models"].items():
            category = data.get("category", "other")

            if category not in models_by_category:
                models_by_category[category] = []

            models_by_category[category].append(
                {
                    "name": name,
                    "params_b": data["params_b"],
                    "hf_id": data.get("hf_id"),
                    "family": data.get("family"),
                }
            )

        return models_by_category

    def get_model_family_variants(self, model_name: str) -> List[str]:
        """
        Get other models from same family, sorted by size (ascending).

        Useful for recommending smaller variants when a model doesn't fit.

        Args:
            model_name: Name of the model

        Returns:
            List of model names in same family, sorted by params (ascending)
        """
        model_info = self.get_model_info(model_name)
        if not model_info or "family" not in model_info:
            return []

        family = model_info["family"]
        variants = []

        for name, data in self.db["models"].items():
            if data.get("family") == family and name != model_name:
                variants.append((name, data["params_b"]))

        # Sort by size (ascending) - smallest first
        variants.sort(key=lambda x: x[1])

        return [v[0] for v in variants]

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with stats like total models, models per category, etc.
        """
        models = self.db["models"]
        categories = {}
        measured_count = 0
        precisions_with_measured = set()

        for name, data in models.items():
            category = data.get("category", "other")
            categories[category] = categories.get(category, 0) + 1

            if "vram" in data:
                measured_count += 1
                precisions_with_measured.update(data["vram"].keys())

        return {
            "total_models": len(models),
            "total_aliases": len(self.db.get("aliases", {})),
            "models_by_category": categories,
            "models_with_measured_vram": measured_count,
            "precisions_with_measured_data": sorted(list(precisions_with_measured)),
        }
