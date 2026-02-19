"""
Compute capability utilities for GPU architecture compatibility checks.

Loads compute_capability.json and provides helpers to check whether
a GPU's SM architecture is supported by a given PyTorch arch list.
"""

import json
import os
import re

# Load compute capability data
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_CC_FILE = os.path.join(_DATA_DIR, "compute_capability.json")

with open(_CC_FILE, "r") as f:
    _CC_DATA = json.load(f)

_COMPUTE_CAPABILITIES = _CC_DATA.get("compute_capabilities", {})


def get_sm_for_compute_capability(cc: str) -> str:
    """
    Convert a compute capability string to an SM identifier.

    Args:
        cc: Compute capability string (e.g., "8.9", "12.0")

    Returns:
        SM identifier (e.g., "sm_89", "sm_120").
        Falls back to generating from CC string if not in database.
    """
    entry = _COMPUTE_CAPABILITIES.get(cc)
    if entry:
        return entry["sm"]

    # Fallback: generate from CC string (e.g., "8.9" -> "sm_89", "12.0" -> "sm_120")
    parts = cc.split(".")
    if len(parts) == 2:
        major, minor = parts
        return f"sm_{major}{minor}"

    return f"sm_{cc.replace('.', '')}"


def get_arch_name(cc: str) -> str:
    """
    Get the human-readable architecture name for a compute capability.

    Args:
        cc: Compute capability string (e.g., "8.9")

    Returns:
        Architecture name (e.g., "Ada Lovelace") or "Unknown" if not found.
    """
    entry = _COMPUTE_CAPABILITIES.get(cc)
    if entry:
        return entry["arch_name"]
    return "Unknown"


def is_sm_in_arch_list(sm: str, arch_list: list) -> bool:
    """
    Check if an SM architecture is supported by a PyTorch arch list.

    Handles three matching strategies:
    1. Direct match: "sm_89" in arch_list
    2. Variant match: "sm_90a" covers "sm_90"
    3. PTX forward compatibility: "compute_XX" covers any sm_YY where YY >= XX
       (PTX is JIT-compiled at runtime for newer architectures)

    Args:
        sm: SM identifier to check (e.g., "sm_120")
        arch_list: PyTorch arch list (e.g., ["sm_50", "sm_90", "compute_90"])

    Returns:
        True if the SM is supported (directly or via PTX forward compatibility)
    """
    if not arch_list:
        return False

    # Extract numeric value from target SM (e.g., "sm_120" -> 120)
    sm_match = re.match(r"sm_(\d+)", sm)
    if not sm_match:
        return False
    target_num = int(sm_match.group(1))

    for entry in arch_list:
        # Direct match: "sm_89" == "sm_89"
        if entry == sm:
            return True

        # Variant match: "sm_90a" covers "sm_90"
        variant_match = re.match(r"sm_(\d+)[a-z]", entry)
        if variant_match and int(variant_match.group(1)) == target_num:
            return True

        # PTX forward compatibility: "compute_90" covers sm_120 (90 <= 120)
        compute_match = re.match(r"compute_(\d+)", entry)
        if compute_match:
            compute_num = int(compute_match.group(1))
            if compute_num <= target_num:
                return True

    return False
