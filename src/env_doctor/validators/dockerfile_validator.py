"""
Dockerfile validator for GPU/CUDA compatibility issues.

This module validates Dockerfiles for common GPU configuration issues such as:
- Using CPU-only base images
- Missing PyTorch --index-url flags
- Installing NVIDIA drivers in containers
- CUDA version mismatches
"""
import os
import re
from typing import List, Tuple, Optional
from pathlib import Path

from .models import (
    ValidationResult,
    Severity,
    is_cpu_only_image,
    is_gpu_enabled_image,
    extract_cuda_version_from_image,
    get_cuda_wheel_suffix,
)


class DockerfileValidator:
    """
    Validates Dockerfiles for GPU/CUDA configuration issues.

    This validator checks for common misconfigurations that prevent
    GPU workloads from running correctly in containers.
    """

    def __init__(self, dockerfile_path: str = "Dockerfile"):
        """
        Initialize the Dockerfile validator.

        Args:
            dockerfile_path: Path to the Dockerfile to validate
        """
        self.dockerfile_path = dockerfile_path
        self.lines: List[str] = []
        self.original_lines: List[str] = []  # Preserve original for line numbers
        self.cuda_version: Optional[str] = None
        self.base_image: Optional[str] = None

    def validate(self) -> ValidationResult:
        """
        Validate the Dockerfile for GPU/CUDA issues.

        Returns:
            ValidationResult: Validation result with all detected issues
        """
        result = ValidationResult(file_path=self.dockerfile_path)

        # Load and preprocess the Dockerfile
        try:
            self._load_dockerfile()
        except FileNotFoundError:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Dockerfile not found: {self.dockerfile_path}",
                recommendation="Ensure the Dockerfile exists at the specified path"
            )
            return result
        except PermissionError:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Permission denied reading: {self.dockerfile_path}",
                recommendation="Check file permissions"
            )
            return result
        except Exception as e:
            result.add_issue(
                line_number=0,
                severity=Severity.ERROR,
                issue=f"Error reading Dockerfile: {str(e)}",
                recommendation="Ensure the file is a valid text file"
            )
            return result

        # Perform all validations
        self._validate_base_image(result)
        self._validate_pip_installs(result)
        self._validate_driver_installation(result)
        self._validate_cuda_toolkit_installation(result)

        # Sort issues by line number
        result.sort_issues()

        return result

    def _load_dockerfile(self) -> None:
        """
        Load and preprocess the Dockerfile.

        Raises:
            FileNotFoundError: If Dockerfile doesn't exist
            PermissionError: If file can't be read
        """
        with open(self.dockerfile_path, 'r', encoding='utf-8') as f:
            self.original_lines = f.readlines()

        # Preprocess: resolve line continuations and strip comments
        self.lines = self._preprocess_dockerfile(self.original_lines)

    def _preprocess_dockerfile(self, lines: List[str]) -> List[str]:
        """
        Preprocess Dockerfile by resolving line continuations and removing comments.

        Args:
            lines: Raw lines from Dockerfile

        Returns:
            Preprocessed lines
        """
        processed = []
        current_line = ""
        current_line_num = 0

        for i, line in enumerate(lines):
            # Remove inline comments (but not in strings)
            if '#' in line:
                # Simple heuristic: remove comments not in quotes
                line = self._remove_inline_comment(line)

            # Handle line continuations
            if line.rstrip().endswith('\\'):
                current_line += line.rstrip()[:-1] + " "
                if not current_line_num:
                    current_line_num = i + 1
            else:
                current_line += line
                if current_line_num:
                    processed.append((current_line_num, current_line.strip()))
                    current_line_num = 0
                else:
                    processed.append((i + 1, current_line.strip()))
                current_line = ""

        return processed

    def _remove_inline_comment(self, line: str) -> str:
        """
        Remove inline comments from a line, preserving quoted strings.

        Args:
            line: Line from Dockerfile

        Returns:
            Line with comment removed
        """
        # Simple approach: split on # and check if we're in quotes
        if '#' not in line:
            return line

        in_quotes = False
        quote_char = None
        for i, char in enumerate(line):
            if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
            elif char == '#' and not in_quotes:
                return line[:i]

        return line

    def _validate_base_image(self, result: ValidationResult) -> None:
        """
        Validate base image selection for GPU support.

        Args:
            result: ValidationResult to add issues to
        """
        from_stages = self._parse_from_statements()

        if not from_stages:
            result.add_issue(
                line_number=0,
                severity=Severity.WARNING,
                issue="No FROM statement found in Dockerfile",
                recommendation="Add a FROM statement with a base image"
            )
            return

        # For multi-stage builds, validate final stage only
        final_stage = from_stages[-1]
        line_num, image = final_stage

        self.base_image = image

        # Check if image is CPU-only
        if is_cpu_only_image(image):
            result.add_issue(
                line_number=line_num,
                severity=Severity.ERROR,
                issue=f"CPU-only base image detected: {image}",
                recommendation="Use a GPU-enabled base image",
                corrected_command="FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04\n"
                                 "# Or: FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime\n"
                                 "# Or: FROM tensorflow/tensorflow:latest-gpu"
            )
            return

        # Check if image is GPU-enabled
        if is_gpu_enabled_image(image):
            # Extract CUDA version
            cuda_ver = extract_cuda_version_from_image(image)
            if cuda_ver:
                self.cuda_version = cuda_ver
                result.add_issue(
                    line_number=line_num,
                    severity=Severity.INFO,
                    issue=f"GPU-enabled base image detected with CUDA {cuda_ver}",
                    recommendation="Ensure pip installations match this CUDA version"
                )
            else:
                result.add_issue(
                    line_number=line_num,
                    severity=Severity.WARNING,
                    issue=f"GPU-enabled base image detected but CUDA version unclear: {image}",
                    recommendation="Use explicit CUDA version in image tag"
                )
        else:
            result.add_issue(
                line_number=line_num,
                severity=Severity.WARNING,
                issue=f"Unknown base image: {image}. Cannot determine GPU support.",
                recommendation="Use a known GPU-enabled base image (nvidia/cuda, pytorch/pytorch, etc.)"
            )

    def _parse_from_statements(self) -> List[Tuple[int, str]]:
        """
        Parse all FROM statements in the Dockerfile.

        Returns:
            List of (line_number, image) tuples
        """
        from_stages = []

        for line_num, line in self.lines:
            if line.upper().startswith('FROM '):
                # Extract image name (handle AS aliases)
                parts = line.split()
                if len(parts) >= 2:
                    image = parts[1]
                    # Remove 'AS alias' if present
                    if 'AS' in [p.upper() for p in parts]:
                        as_idx = [p.upper() for p in parts].index('AS')
                        image = parts[1] if as_idx > 1 else parts[1]
                    from_stages.append((line_num, image))

        return from_stages

    def _validate_pip_installs(self, result: ValidationResult) -> None:
        """
        Validate pip install commands for GPU libraries.

        Args:
            result: ValidationResult to add issues to
        """
        for line_num, line in self.lines:
            # Look for RUN pip install commands
            if not ('pip install' in line.lower() and line.upper().startswith('RUN')):
                continue

            # Skip requirements file installs
            if '-r requirements.txt' in line or '-r requirements' in line:
                continue

            # Check for PyTorch installation
            if 'torch' in line and 'torchvision' not in line:
                self._validate_pytorch_install(line_num, line, result)
            elif 'torch' in line and 'torchvision' in line:
                self._validate_pytorch_install(line_num, line, result)

            # Check for TensorFlow installation
            if 'tensorflow' in line.lower() and 'tensorflow[' not in line.lower():
                self._validate_tensorflow_install(line_num, line, result)

    def _validate_pytorch_install(self, line_num: int, line: str, result: ValidationResult) -> None:
        """
        Validate PyTorch installation command.

        Args:
            line_num: Line number
            line: Installation command
            result: ValidationResult to add issues to
        """
        # Check if --index-url is present
        if '--index-url' not in line:
            # Get the correct wheel suffix based on detected CUDA version
            corrected_cmd = None
            if self.cuda_version:
                wheel_suffix = get_cuda_wheel_suffix(self.cuda_version)
                if wheel_suffix:
                    corrected_cmd = f"RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/{wheel_suffix}"

            result.add_issue(
                line_number=line_num,
                severity=Severity.ERROR,
                issue="PyTorch installation missing --index-url flag",
                recommendation="Add --index-url to install the correct CUDA version. "
                             f"For CUDA {self.cuda_version or 'X.Y'}, use --index-url https://download.pytorch.org/whl/{get_cuda_wheel_suffix(self.cuda_version) if self.cuda_version else 'cuXXX'}",
                corrected_command=corrected_cmd
            )
        elif self.cuda_version:
            # Check if the CUDA version matches
            wheel_suffix = get_cuda_wheel_suffix(self.cuda_version)
            if wheel_suffix and wheel_suffix not in line:
                result.add_issue(
                    line_number=line_num,
                    severity=Severity.WARNING,
                    issue=f"PyTorch CUDA version mismatch. Base image uses CUDA {self.cuda_version}",
                    recommendation=f"Use --index-url with '{wheel_suffix}' to match base image CUDA version",
                    corrected_command=f"RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/{wheel_suffix}"
                )

    def _validate_tensorflow_install(self, line_num: int, line: str, result: ValidationResult) -> None:
        """
        Validate TensorFlow installation command.

        Args:
            line_num: Line number
            line: Installation command
            result: ValidationResult to add issues to
        """
        # Check if using tensorflow or tensorflow-gpu (deprecated)
        if 'tensorflow-gpu' in line.lower():
            result.add_issue(
                line_number=line_num,
                severity=Severity.WARNING,
                issue="tensorflow-gpu is deprecated",
                recommendation="Use 'tensorflow[and-cuda]' or 'tensorflow' (GPU support is automatic with CUDA)",
                corrected_command="RUN pip install tensorflow[and-cuda]"
            )
        elif 'tensorflow[and-cuda]' not in line.lower() and 'tensorflow-gpu' not in line.lower():
            result.add_issue(
                line_number=line_num,
                severity=Severity.WARNING,
                issue="TensorFlow installation may not include GPU support",
                recommendation="Use 'tensorflow[and-cuda]' to ensure GPU support",
                corrected_command="RUN pip install tensorflow[and-cuda]"
            )

    def _validate_driver_installation(self, result: ValidationResult) -> None:
        """
        Detect and flag NVIDIA driver installations (which should never be in containers).

        Args:
            result: ValidationResult to add issues to
        """
        for line_num, line in self.lines:
            # Look for apt/yum/dnf commands
            if not line.upper().startswith('RUN'):
                continue

            line_lower = line.lower()

            # Check for NVIDIA driver installation using regex for more flexible matching
            import re
            driver_patterns = [
                r'apt-get\s+install.*nvidia-driver',
                r'apt\s+install.*nvidia-driver',
                r'yum\s+install.*nvidia-driver',
                r'dnf\s+install.*nvidia-driver',
            ]

            for pattern in driver_patterns:
                if re.search(pattern, line_lower):
                    result.add_issue(
                        line_number=line_num,
                        severity=Severity.ERROR,
                        issue="NVIDIA drivers must NOT be installed in containers",
                        recommendation="Remove driver installation. Drivers must be installed on the host system, not in containers.",
                        corrected_command="# Remove this line - drivers are provided by the host"
                    )
                    break

    def _validate_cuda_toolkit_installation(self, result: ValidationResult) -> None:
        """
        Flag CUDA toolkit installations that may be unnecessary.

        Args:
            result: ValidationResult to add issues to
        """
        import re

        # Keywords that suggest compilation requirements
        compilation_keywords = ['flash-attn', 'flash_attn', 'xformers', 'auto-gptq', 'nvcc', 'gcc', 'g++', 'build-essential']

        # Check if Dockerfile needs CUDA toolkit for compilation
        needs_compilation = False
        for line_num, line in self.lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in compilation_keywords):
                needs_compilation = True
                break

        # Check for CUDA toolkit installation
        for line_num, line in self.lines:
            if not line.upper().startswith('RUN'):
                continue

            line_lower = line.lower()

            # Check for CUDA toolkit installation using regex
            toolkit_patterns = [
                r'apt-get\s+install.*cuda-toolkit',
                r'apt\s+install.*cuda-toolkit',
                r'apt-get\s+install.*\bcuda\b',
                r'apt\s+install.*\bcuda\b',
                r'yum\s+install.*\bcuda\b',
                r'dnf\s+install.*\bcuda\b',
            ]

            # Skip if it's just libcuda
            if 'libcuda' in line_lower and 'cuda-toolkit' not in line_lower:
                continue

            for pattern in toolkit_patterns:
                if re.search(pattern, line_lower):
                    if needs_compilation:
                        result.add_issue(
                            line_number=line_num,
                            severity=Severity.INFO,
                            issue="CUDA toolkit installation detected",
                            recommendation="Toolkit appears needed for compilation (flash-attention, xformers, etc.). "
                                         "This adds 2-5GB to image size. Consider using pre-built wheels if available."
                        )
                    else:
                        result.add_issue(
                            line_number=line_num,
                            severity=Severity.WARNING,
                            issue="CUDA toolkit installation may be unnecessary",
                            recommendation="Runtime-only containers don't need the full toolkit (adds 2-5GB). "
                                         "Only install if compiling CUDA extensions (flash-attention, xformers, etc.)",
                            corrected_command="# Remove if not compiling CUDA code"
                        )
                    break
