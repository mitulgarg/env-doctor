"""
Compatibility database wrapper for Dockerfile validation.

This module provides a testable interface to query env-doctor's CUDA
compatibility database for verified install commands.
"""
from typing import Optional, Dict, List


class CompatibilityDB:
    """
    Wrapper for env-doctor's compatibility database.

    Provides methods to query DB-verified install commands for GPU libraries
    based on CUDA version. "Verified" means present in env-doctor's
    compatibility database.
    """

    def __init__(self, data: Optional[Dict] = None):
        """
        Initialize the compatibility database.

        Args:
            data: Optional dict to use instead of loading from file.
                  If None, loads from env_doctor.db module.
        """
        if data is not None:
            self.data = data
        else:
            # Import here to avoid circular dependency
            from env_doctor.db import DB_DATA
            self.data = DB_DATA

        self.recommendations = self.data.get("recommendations", {})

    def get_verified_install_command(self, cuda_mm: str, lib: str) -> Optional[str]:
        """
        Get the DB-verified install command for a library and CUDA version.

        Args:
            cuda_mm: CUDA version in major.minor format (e.g., "12.1", "11.8")
            lib: Library name (e.g., "torch", "tensorflow", "jax")

        Returns:
            Full pip install command string if verified in DB, None otherwise
        """
        if cuda_mm not in self.recommendations:
            return None

        return self.recommendations[cuda_mm].get(lib)

    def has_cuda_entry(self, cuda_mm: str) -> bool:
        """
        Check if the DB has any entries for a CUDA version.

        Args:
            cuda_mm: CUDA version in major.minor format (e.g., "12.1")

        Returns:
            True if DB has recommendations for this CUDA version
        """
        return cuda_mm in self.recommendations

    def all_cuda_versions(self) -> List[str]:
        """
        Get all CUDA versions in the database.

        Returns:
            List of CUDA version strings (e.g., ["12.1", "11.8", ...])
        """
        return list(self.recommendations.keys())

    def find_cuda_versions_for_library_version(self, lib: str, version: str) -> List[str]:
        """
        Find CUDA versions that have a specific library version in DB.

        Searches for install commands containing the exact library version pin.

        Args:
            lib: Library name (e.g., "torch", "tensorflow")
            version: Version string (e.g., "2.0.1", "2.15.0")

        Returns:
            List of CUDA versions where this library version is verified
        """
        pin_pattern = f"{lib}=={version}"
        matching_cudas = []

        for cuda_version, libs in self.recommendations.items():
            cmd = libs.get(lib, "")
            # Check for exact match with word boundaries
            if pin_pattern in cmd:
                # Verify it's a standalone match (not part of a longer version)
                # by checking it's followed by space, end of string, or -
                idx = cmd.find(pin_pattern)
                if idx != -1:
                    end_idx = idx + len(pin_pattern)
                    if end_idx >= len(cmd) or cmd[end_idx] in [' ', '\n', '+']:
                        matching_cudas.append(cuda_version)

        return matching_cudas
