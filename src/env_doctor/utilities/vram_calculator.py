"""
VRAM calculation utilities for AI models.

Provides three-tier hybrid approach:
1. Measured VRAM values (when available in database) - most accurate
2. HuggingFace Hub API (fetch model parameters dynamically) - accurate for any HF model
3. Formula-based estimation (parameter-based calculation) - fallback for any model
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# HuggingFace Hub integration (optional dependency)
try:
    from huggingface_hub import model_info, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Define dummy exception classes for type checking when HF not available
    class RepositoryNotFoundError(Exception):
        pass
    class HfHubHTTPError(Exception):
        pass
    model_info = None
    HfApi = None


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

    # HuggingFace API timeout (seconds)
    HF_TIMEOUT = 5

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

        self.db_path = db_path

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

        # Initialize hf_cache section if not present
        if "hf_cache" not in self.db:
            self.db["hf_cache"] = {}

    def calculate_vram(
        self, model_name: str, precision: str = "fp16"
    ) -> Dict[str, Any]:
        """
        Calculate VRAM requirement for a model using 3-tier fallback.

        Strategy:
        1. Check if model exists in local database (models section)
        2. Check if model exists in hf_cache
        3. If not found, try fetching from HuggingFace API
        4. Use measured VRAM if available, otherwise estimate from params

        Args:
            model_name: Name of the model (e.g., "llama-3-8b" or "meta-llama/Llama-2-7b-hf")
            precision: Precision level (fp32, fp16, bf16, int8, int4, fp8)

        Returns:
            Dict with keys:
                - vram_mb: Required VRAM in MB
                - source: "measured", "estimated", or "huggingface_api"
                - params_b: Model parameter count in billions
                - formula (if estimated): The calculation formula used
                - fetched_from_hf (if applicable): True if data came from HF API

        Raises:
            ValueError: If model not found anywhere
            KeyError: If precision not supported
        """
        # Normalize model name (lowercase, resolve aliases)
        normalized_name = self._normalize_model_name(model_name)
        model_data = None
        fetched_from_hf = False

        # Tier 1: Check local database (models section)
        if normalized_name in self.db["models"]:
            model_data = self.db["models"][normalized_name]

        # Tier 2: Check hf_cache
        elif normalized_name in self.db.get("hf_cache", {}):
            model_data = self.db["hf_cache"][normalized_name]
            fetched_from_hf = True

        # Tier 3: Try fetching from HuggingFace API
        else:
            # Try the model_name as a HuggingFace ID directly
            hf_result = self._fetch_from_huggingface(model_name)

            if hf_result:
                model_data = hf_result
                fetched_from_hf = True
                # Cache the result for future use
                self._save_to_cache(normalized_name, model_data)

        # If still not found, raise error
        if model_data is None:
            hf_note = ""
            if HF_AVAILABLE:
                hf_note = " Also tried HuggingFace API but model was not found."
            else:
                hf_note = " Install 'huggingface_hub' to enable automatic model lookup."

            raise ValueError(
                f"Model '{model_name}' not found in database.{hf_note} "
                f"Use 'env-doctor model --list' to see available models."
            )

        params_b = model_data["params_b"]

        # Try measured VRAM first (highest accuracy) - only available for local DB models
        if "vram" in model_data and precision in model_data["vram"]:
            result = {
                "vram_mb": model_data["vram"][precision],
                "source": "measured",
                "params_b": params_b,
            }
            if fetched_from_hf:
                result["fetched_from_hf"] = True
            return result

        # Fallback to formula-based calculation
        vram_mb = self._calculate_from_params(params_b, precision)

        result = {
            "vram_mb": vram_mb,
            "source": "huggingface_api" if fetched_from_hf else "estimated",
            "params_b": params_b,
            "formula": f"{params_b}B × {self.BYTES_PER_PARAM[precision]} bytes/param × {self.OVERHEAD_MULTIPLIER} overhead",
        }
        if fetched_from_hf:
            result["fetched_from_hf"] = True
        return result

    def calculate_all_precisions(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Calculate VRAM for all supported precisions.

        Args:
            model_name: Name of the model

        Returns:
            Dict mapping precision -> vram_info
            Example: {"fp16": {...}, "int8": {...}, "int4": {...}}

        Raises:
            ValueError: If model not found in database or HuggingFace
        """
        results = {}
        model_not_found_error = None

        for precision in ["fp32", "fp16", "bf16", "int8", "int4", "fp8"]:
            try:
                results[precision] = self.calculate_vram(model_name, precision)
            except KeyError:
                # Skip precisions that aren't supported
                pass
            except ValueError as e:
                # Store model not found error to raise later if no results
                model_not_found_error = e

        # If no results and we had a model not found error, raise it
        if not results and model_not_found_error:
            raise model_not_found_error

        return results

    def _fetch_from_huggingface(self, hf_model_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch model parameters from HuggingFace Hub API.

        Args:
            hf_model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")

        Returns:
            Dict with params_b and metadata if successful, None if failed
        """
        if not HF_AVAILABLE:
            return None

        try:
            # Fetch model info from HuggingFace Hub
            info = model_info(hf_model_id, timeout=self.HF_TIMEOUT)

            params_b = None

            # Try to get parameter count from safetensors metadata (most accurate)
            if hasattr(info, 'safetensors') and info.safetensors:
                if 'total' in info.safetensors:
                    # Convert to billions
                    params_b = info.safetensors['total'] / 1_000_000_000

            # Fallback: try to get from model card metadata
            if params_b is None and hasattr(info, 'card_data') and info.card_data:
                card = info.card_data
                if hasattr(card, 'model_index') and card.model_index:
                    for model in card.model_index:
                        if 'results' in model:
                            # Some models have parameter info in results
                            pass

            # Fallback: try to extract from model name (e.g., "7b", "13b")
            if params_b is None:
                import re
                match = re.search(r'(\d+\.?\d*)b', hf_model_id.lower())
                if match:
                    params_b = float(match.group(1))

            if params_b is None:
                return None

            return {
                "params_b": params_b,
                "hf_id": hf_model_id,
                "source": "huggingface_api",
            }

        except (RepositoryNotFoundError, HfHubHTTPError):
            return None
        except Exception:
            # Catch any other errors (timeout, network issues, etc.)
            return None

    def _save_to_cache(self, model_name: str, model_data: Dict[str, Any]) -> None:
        """
        Save fetched model data to hf_cache in model_requirements.json.

        Args:
            model_name: Normalized model name to use as key
            model_data: Model data dict with params_b, hf_id, etc.
        """
        try:
            self.db["hf_cache"][model_name] = model_data
            with open(self.db_path, "w") as f:
                json.dump(self.db, f, indent=2)
        except (IOError, OSError):
            # Silently fail if we can't write to cache
            pass

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
        Get full model information from database or hf_cache.

        Args:
            model_name: Name of the model

        Returns:
            Model data dict, or None if not found
        """
        normalized_name = self._normalize_model_name(model_name)

        # Check local models first
        if normalized_name in self.db["models"]:
            return self.db["models"][normalized_name]

        # Check hf_cache
        if normalized_name in self.db.get("hf_cache", {}):
            return self.db["hf_cache"][normalized_name]

        return None

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
