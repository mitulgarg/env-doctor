"""
Validation models for Dockerfile and docker-compose validators.

This module defines the data structures used to represent validation issues
and results for container configuration files.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Severity(Enum):
    """Severity level of a validation issue."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# Known CPU-only base images that won't work with GPU workloads
CPU_ONLY_IMAGES = {
    "python",
    "ubuntu",
    "debian",
    "alpine",
    "node",
    "ruby",
    "java",
    "openjdk",
    "golang",
    "rust",
}

# Known GPU-enabled base images
GPU_ENABLED_IMAGES = {
    "nvidia/cuda",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "nvcr.io",  # NVIDIA NGC container registry
}

# CUDA version to pip wheel suffix mapping
CUDA_TO_WHEEL = {
    "11.7": "cu117",
    "11.8": "cu118",
    "12.0": "cu120",
    "12.1": "cu121",
    "12.2": "cu122",
    "12.3": "cu123",
    "12.4": "cu124",
}


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue found in a container configuration file.

    Attributes:
        line_number: Line number where the issue was found (1-indexed)
        severity: Severity level (error, warning, info)
        issue: Description of the issue
        recommendation: Suggested fix or improvement
        corrected_command: Optional corrected command/configuration
    """
    line_number: int
    severity: Severity
    issue: str
    recommendation: str
    corrected_command: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of validating a container configuration file.

    Attributes:
        file_path: Path to the validated file
        issues: List of validation issues found
        error_count: Number of errors found
        warning_count: Number of warnings found
        info_count: Number of info messages
        success: Whether validation passed (no errors)
    """
    file_path: str
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of info-level issues."""
        return sum(1 for issue in self.issues if issue.severity == Severity.INFO)

    @property
    def success(self) -> bool:
        """Returns True if no errors were found."""
        return self.error_count == 0

    def add_issue(
        self,
        line_number: int,
        severity: Severity,
        issue: str,
        recommendation: str,
        corrected_command: Optional[str] = None
    ) -> None:
        """
        Add a validation issue to the result.

        Args:
            line_number: Line number where issue was found
            severity: Severity level
            issue: Issue description
            recommendation: Suggested fix
            corrected_command: Optional corrected command
        """
        self.issues.append(ValidationIssue(
            line_number=line_number,
            severity=severity,
            issue=issue,
            recommendation=recommendation,
            corrected_command=corrected_command
        ))

    def get_issues_by_severity(self, severity: Severity) -> List[ValidationIssue]:
        """
        Get all issues of a specific severity level.

        Args:
            severity: Severity level to filter by

        Returns:
            List of issues with the specified severity
        """
        return [issue for issue in self.issues if issue.severity == severity]

    def sort_issues(self) -> None:
        """Sort issues by line number, then by severity."""
        severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        self.issues.sort(key=lambda x: (x.line_number, severity_order[x.severity]))

def get_cuda_wheel_suffix(cuda_version: str) -> Optional[str]:
    """
    Get the pip wheel suffix for a given CUDA version.

    Args:
        cuda_version: CUDA version string (e.g., "12.1.0", "11.8")

    Returns:
        Wheel suffix (e.g., "cu121") or None if version not found
    """
    # Extract major.minor version
    parts = cuda_version.split(".")
    if len(parts) >= 2:
        major_minor = f"{parts[0]}.{parts[1]}"
        return CUDA_TO_WHEEL.get(major_minor)
    return None


def is_cpu_only_image(image: str) -> bool:
    """
    Check if a Docker image is CPU-only (no GPU support).

    Args:
        image: Docker image name (e.g., "python:3.10", "ubuntu:22.04")

    Returns:
        True if the image is CPU-only
    """
    image_lower = image.lower()
    for cpu_image in CPU_ONLY_IMAGES:
        if image_lower.startswith(cpu_image + ":") or image_lower == cpu_image:
            return True
    return False


def is_gpu_enabled_image(image: str) -> bool:
    """
    Check if a Docker image has GPU support.

    Args:
        image: Docker image name (e.g., "nvidia/cuda:12.1.0-runtime")

    Returns:
        True if the image has GPU support
    """
    image_lower = image.lower()

    # Check against known GPU-enabled prefixes
    for gpu_image in GPU_ENABLED_IMAGES:
        if image_lower.startswith(gpu_image):
            # Additional check for TensorFlow - must have 'gpu' in tag
            if "tensorflow" in image_lower:
                return "gpu" in image_lower
            return True

    return False


def extract_cuda_version_from_image(image: str) -> Optional[str]:
    """
    Extract CUDA version from a Docker image tag.

    Args:
        image: Docker image name with tag (e.g., "nvidia/cuda:12.1.0-runtime-ubuntu22.04")

    Returns:
        CUDA version string (e.g., "12.1") or None if not found
    """
    if ":" not in image:
        return None

    tag = image.split(":")[1]

    # Look for pattern like "12.1.0" or "11.8"
    import re
    match = re.match(r"(\d+\.\d+)(?:\.\d+)?", tag)
    if match:
        return match.group(1)

    return None
