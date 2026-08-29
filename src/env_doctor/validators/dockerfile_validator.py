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


# Keywords that indicate CUDA compilation happens in a stage.
# Used to decide whether a -devel base image is actually justified.
COMPILATION_KEYWORDS = [
    'flash-attn', 'flash_attn', 'xformers', 'auto-gptq',
    'nvcc', 'gcc', 'g++', 'build-essential',
]

# Patterns that make a stage's dependency set opaque, so we cannot safely
# conclude that no compilation happens.
OPAQUE_DEPENDENCY_PATTERNS = [
    r'pip\s+install\b.*\s-r\s',
    r'pip\s+install\s.*--no-build-isolation',
    r'pip\s+install\s.*\bgit\+',
    r'pip\s+install\s+\.\s*$',
    r'python\s+setup\.py',
]


class DockerfileValidator:
    """
    Validates Dockerfiles for GPU/CUDA configuration issues.

    This validator checks for common misconfigurations that prevent
    GPU workloads from running correctly in containers.
    """

    def __init__(self, dockerfile_path: str = "Dockerfile", compat_db=None):
        """
        Initialize the Dockerfile validator.

        Args:
            dockerfile_path: Path to the Dockerfile to validate
            compat_db: Optional CompatibilityDB instance for testing
        """
        self.dockerfile_path = dockerfile_path
        self.lines: List[str] = []
        self.original_lines: List[str] = []  # Preserve original for line numbers
        self.cuda_version: Optional[str] = None
        self.base_image: Optional[str] = None
        self.base_image_flavor: Optional[str] = None  # "runtime", "devel", or None
        self.base_image_variant: Optional[str] = None  # base/runtime/devel/cudnn-runtime/cudnn-devel
        self.stages: List[dict] = []  # One entry per FROM statement
        self.final_stage_lines: List[Tuple[int, str]] = []  # Lines belonging to the final stage
        self.detected_libraries: dict = {}  # {lib_name: {version, line_number, index_url, ...}}

        # Initialize compatibility DB
        if compat_db is not None:
            self.compat_db = compat_db
        else:
            from .compat_db import CompatibilityDB
            self.compat_db = CompatibilityDB()

        # Load pip package deprecations from migrations.json
        self.deprecated_packages = self._load_deprecated_packages()

        # Load approximate image sizes for bloat estimates
        self.image_sizes = self._load_image_sizes()

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

        # Partition lines by build stage before any stage-sensitive validation
        self._partition_lines_by_stage()

        # Perform all validations (order matters - base image and pip parsing must come first)
        self._validate_base_image(result)
        self._validate_pip_installs(result)  # Populates self.detected_libraries
        self._validate_cpu_base_with_gpu_libs(result)  # Must run after pip parsing
        self._validate_library_cuda_compatibility(result)  # Uses detected_libraries and cuda_version
        self._validate_multi_library_compatibility(result)  # Multi-lib summary
        self._validate_runtime_devel_mismatch(result)  # Check runtime vs devel
        self._validate_driver_installation(result)
        self._validate_cuda_toolkit_installation(result)
        self._validate_image_bloat(result)  # devel image without compilation
        self._suggest_multi_stage_build(result)  # single-stage devel that must compile

        # Sort issues by line number
        result.sort_issues()

        return result

    def _load_deprecated_packages(self) -> dict:
        """
        Load deprecated pip packages from migrations.json.

        Returns:
            Dict mapping deprecated package names to their replacement info
        """
        import json
        import os

        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            migrations_path = os.path.join(base_path, "..", "data", "migrations.json")

            with open(migrations_path, "r") as f:
                migrations_data = json.load(f)

            return migrations_data.get("pip_packages", {})
        except Exception:
            # If migrations.json doesn't exist or can't be loaded, return empty dict
            return {}

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
        from_stages = self.stages or self._parse_from_statements()

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
        line_num = final_stage["line_number"]
        image = final_stage["image"]

        self.base_image = image

        # Extract base image flavor (runtime/devel)
        image_lower = image.lower()
        if "-runtime" in image_lower:
            self.base_image_flavor = "runtime"
        elif "-devel" in image_lower:
            self.base_image_flavor = "devel"
        else:
            self.base_image_flavor = None

        # Finer-grained variant used for size estimates
        self.base_image_variant = self._detect_image_variant(image)

        # Check if image is GPU-enabled (defer CPU-only error to later validation)
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

    def _parse_from_statements(self) -> List[dict]:
        """
        Parse all FROM statements in the Dockerfile.

        Returns:
            List of dicts with keys: line_number, image, alias, index
        """
        from_stages = []

        for line_num, line in self.lines:
            if not line.upper().startswith('FROM '):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            image = parts[1]

            # Capture 'AS <alias>' if present
            alias = None
            upper_parts = [p.upper() for p in parts]
            if 'AS' in upper_parts:
                as_idx = upper_parts.index('AS')
                if as_idx + 1 < len(parts):
                    alias = parts[as_idx + 1]

            from_stages.append({
                "line_number": line_num,
                "image": image,
                "alias": alias,
                "index": len(from_stages),
            })

        return from_stages

    def _partition_lines_by_stage(self) -> None:
        """
        Assign each preprocessed line to the build stage that contains it.

        Populates self.stages (each with a "lines" list) and
        self.final_stage_lines. For a single-stage Dockerfile,
        self.final_stage_lines is equivalent to self.lines, so stage-scoped
        checks behave exactly as before.
        """
        self.stages = self._parse_from_statements()

        if not self.stages:
            self.final_stage_lines = list(self.lines)
            return

        for stage in self.stages:
            stage["lines"] = []

        current = None
        for line_num, line in self.lines:
            # A FROM line opens the next stage
            for stage in self.stages:
                if stage["line_number"] == line_num:
                    current = stage
                    break
            if current is not None:
                current["lines"].append((line_num, line))

        self.final_stage_lines = self.stages[-1]["lines"]

    def _detect_compilation(self, lines: List[Tuple[int, str]]) -> Tuple[bool, int]:
        """
        Detect whether the given lines perform CUDA compilation.

        Args:
            lines: (line_number, text) tuples to scan

        Returns:
            (needs_compilation, line_number) - line_number is 0 when not found
        """
        for line_num, line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in COMPILATION_KEYWORDS):
                return True, line_num
        return False, 0

    def _has_opaque_dependencies(self, lines: List[Tuple[int, str]]) -> bool:
        """
        Check whether a stage installs dependencies we cannot inspect.

        When a stage installs from requirements.txt, a local project, or a git
        URL, we cannot know whether it compiles CUDA extensions, so downgrade
        suggestions must be suppressed.

        Args:
            lines: (line_number, text) tuples to scan

        Returns:
            True if dependencies are opaque
        """
        for _, line in lines:
            line_lower = line.lower()
            for pattern in OPAQUE_DEPENDENCY_PATTERNS:
                if re.search(pattern, line_lower):
                    return True
        return False

    def _detect_image_variant(self, image: str) -> Optional[str]:
        """
        Classify a CUDA image into a size variant.

        Args:
            image: Full base image reference

        Returns:
            One of "base", "runtime", "devel", "cudnn-runtime", "cudnn-devel",
            or None when the variant cannot be determined.
        """
        if not image:
            return None

        image_lower = image.lower()
        has_cudnn = 'cudnn' in image_lower

        if '-devel' in image_lower or image_lower.endswith('devel'):
            return "cudnn-devel" if has_cudnn else "devel"
        if '-runtime' in image_lower or image_lower.endswith('runtime'):
            return "cudnn-runtime" if has_cudnn else "runtime"
        if '-base' in image_lower or image_lower.endswith('base'):
            return "base"
        return None

    def _load_image_sizes(self) -> dict:
        """
        Load approximate CUDA image sizes from cuda_image_sizes.json.

        Returns:
            Parsed size data, or an empty dict if unavailable
        """
        import json

        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            sizes_path = os.path.join(base_path, "..", "data", "cuda_image_sizes.json")

            with open(sizes_path, "r") as f:
                return json.load(f)
        except Exception:
            # Missing or unreadable data file degrades to "no size estimate"
            return {}

    def _format_savings(self, from_variant: str, to_variant: str) -> str:
        """
        Build a human-readable size delta between two image variants.

        Deltas are quoted rather than absolute sizes: the ratio between
        variants is stable across CUDA releases, the absolute megabytes are not.

        Args:
            from_variant: Current variant
            to_variant: Suggested variant

        Returns:
            A phrase like "roughly 3.3 GB smaller", or "" if sizes are unknown
        """
        variants = self.image_sizes.get("variants", {})
        current = variants.get(from_variant, {}).get("approx_gb")
        target = variants.get(to_variant, {}).get("approx_gb")

        if current is None or target is None:
            return ""

        delta = current - target
        if delta <= 0:
            return ""

        return f"roughly {delta:.1f} GB smaller"

    def _parse_pip_install_line(self, line: str) -> dict:
        """
        Parse a pip install command line into structured data.

        Args:
            line: Preprocessed pip install command line

        Returns:
            Dict with:
                - packages: List of {name, version, raw_token}
                - index_url: str or None
                - extra_index_url: str or None
                - find_links: str or None
                - other_flags: List[str]
        """
        result = {
            "packages": [],
            "index_url": None,
            "extra_index_url": None,
            "find_links": None,
            "other_flags": []
        }

        tokens = line.split()
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Check for flags
            if token == "--index-url" and i + 1 < len(tokens):
                result["index_url"] = tokens[i + 1]
                i += 2
                continue
            elif token == "--extra-index-url" and i + 1 < len(tokens):
                result["extra_index_url"] = tokens[i + 1]
                i += 2
                continue
            elif token in ["-f", "--find-links"] and i + 1 < len(tokens):
                result["find_links"] = tokens[i + 1]
                i += 2
                continue
            elif token in ["-r", "--requirement"]:
                # Skip requirements file entries
                i += 2 if i + 1 < len(tokens) else 1
                continue
            elif token.startswith("-"):
                result["other_flags"].append(token)
                i += 1
                continue
            elif token.upper() in ["RUN", "PIP", "INSTALL", "PIP3", "PYTHON", "PYTHON3"]:
                # Skip command keywords
                i += 1
                continue

            # Parse package name and version
            raw_token = token
            if "==" in token:
                name, version = token.split("==", 1)
                # Handle extras like torch[cuda]
                if "[" in name:
                    name = name.split("[")[0]
            else:
                name = token
                if "[" in name:
                    name = name.split("[")[0]
                version = None

            # Only track GPU-related libraries
            if name.lower() in ["torch", "torchvision", "torchaudio", "tensorflow", "jax", "tensorflow-gpu"]:
                result["packages"].append({
                    "name": name.lower(),
                    "version": version,
                    "raw_token": raw_token
                })

            i += 1

        return result

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

            # Parse the pip install line
            parsed = self._parse_pip_install_line(line)

            # Update detected libraries (last occurrence wins)
            for pkg in parsed["packages"]:
                lib_name = pkg["name"]
                self.detected_libraries[lib_name] = {
                    "version": pkg["version"],
                    "line_number": line_num,
                    "index_url": parsed["index_url"],
                    "extra_index_url": parsed["extra_index_url"],
                    "find_links": parsed["find_links"],
                    "raw_token": pkg["raw_token"]
                }

            # Check for deprecated packages
            self._check_deprecated_packages(line_num, line, parsed, result)

            # Check for PyTorch installation
            if 'torch' in line:
                self._validate_pytorch_install(line_num, line, result)

            # Check for TensorFlow installation
            if 'tensorflow' in line.lower():
                self._validate_tensorflow_install(line_num, line, result)

    def _check_deprecated_packages(self, line_num: int, line: str, parsed: dict, result: ValidationResult) -> None:
        """
        Check for deprecated pip packages and suggest replacements.

        Args:
            line_num: Line number
            line: Full install command line
            parsed: Parsed pip install data
            result: ValidationResult to add issues to
        """
        if not self.deprecated_packages:
            return

        for pkg in parsed["packages"]:
            pkg_name = pkg["name"]

            if pkg_name in self.deprecated_packages:
                deprecation_info = self.deprecated_packages[pkg_name]
                replacement = deprecation_info.get("replacement", pkg_name)
                reason = deprecation_info.get("reason", f"{pkg_name} is deprecated")

                # Build corrected command by replacing the deprecated package
                corrected_line = line

                # Replace the package token in the line
                old_token = pkg["raw_token"]
                new_token = replacement

                # If there was a pinned version and we have a recommended version, use it
                if "recommended_version" in deprecation_info and "==" not in replacement:
                    new_token = f"{replacement}=={deprecation_info['recommended_version']}"

                corrected_line = corrected_line.replace(old_token, new_token, 1)

                result.add_issue(
                    line_number=line_num,
                    severity=Severity.WARNING,
                    issue=f"{pkg_name} is deprecated",
                    recommendation=f"{reason}. Use '{replacement}' instead.",
                    corrected_command=corrected_line
                )

    def _validate_pytorch_install(self, line_num: int, line: str, result: ValidationResult) -> None:
        """
        Validate PyTorch installation command.

        Args:
            line_num: Line number
            line: Installation command
            result: ValidationResult to add issues to
        """
        # Check if --index-url or similar is present
        if '--index-url' not in line and '--extra-index-url' not in line and '-f' not in line and '--find-links' not in line:
            # Get DB-verified install command if CUDA version is known
            corrected_cmd = None
            recommendation = "Add --index-url to install the correct CUDA version."

            if self.cuda_version:
                # Try to get DB-verified command
                verified_cmd = self.compat_db.get_verified_install_command(self.cuda_version, "torch")
                if verified_cmd:
                    corrected_cmd = f"RUN {verified_cmd}"
                    recommendation = f"For CUDA {self.cuda_version}, use the verified install command from env-doctor's compatibility database."
                else:
                    # Fallback to generic mapping
                    wheel_suffix = get_cuda_wheel_suffix(self.cuda_version)
                    if wheel_suffix:
                        corrected_cmd = f"RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/{wheel_suffix}"
                        recommendation = f"For CUDA {self.cuda_version}, use --index-url https://download.pytorch.org/whl/{wheel_suffix}"

            result.add_issue(
                line_number=line_num,
                severity=Severity.ERROR,
                issue="PyTorch installation missing --index-url flag",
                recommendation=recommendation,
                corrected_command=corrected_cmd
            )
        elif self.cuda_version:
            # Check if the CUDA version matches
            wheel_suffix = get_cuda_wheel_suffix(self.cuda_version)
            if wheel_suffix and wheel_suffix not in line:
                # Get DB-verified command for suggestion
                verified_cmd = self.compat_db.get_verified_install_command(self.cuda_version, "torch")
                corrected_cmd = None
                if verified_cmd:
                    corrected_cmd = f"RUN {verified_cmd}"

                result.add_issue(
                    line_number=line_num,
                    severity=Severity.WARNING,
                    issue=f"PyTorch CUDA version mismatch. Base image uses CUDA {self.cuda_version}",
                    recommendation=f"Use --index-url with '{wheel_suffix}' to match base image CUDA version",
                    corrected_command=corrected_cmd if corrected_cmd else f"RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/{wheel_suffix}"
                )

    def _validate_tensorflow_install(self, line_num: int, line: str, result: ValidationResult) -> None:
        """
        Validate TensorFlow installation command.

        Args:
            line_num: Line number
            line: Installation command
            result: ValidationResult to add issues to
        """
        # Skip if tensorflow-gpu is present - already handled by deprecation system
        if 'tensorflow-gpu' in line.lower():
            return

        # Check if using tensorflow without GPU support
        if 'tensorflow[and-cuda]' not in line.lower():
            result.add_issue(
                line_number=line_num,
                severity=Severity.WARNING,
                issue="TensorFlow installation may not include GPU support",
                recommendation="Use 'tensorflow[and-cuda]' to ensure GPU support",
                corrected_command="RUN pip install tensorflow[and-cuda]"
            )

    def _validate_library_cuda_compatibility(self, result: ValidationResult) -> None:
        """
        Validate that installed library versions match DB-verified combinations for the base CUDA.

        Args:
            result: ValidationResult to add issues to
        """
        if not self.detected_libraries:
            return

        # Focus on main GPU libraries
        main_libs = ["torch", "tensorflow", "jax"]

        for lib in main_libs:
            if lib not in self.detected_libraries:
                continue

            lib_info = self.detected_libraries[lib]
            line_num = lib_info["line_number"]
            pinned_version = lib_info["version"]

            # Skip if base CUDA is unknown
            if not self.cuda_version:
                result.add_issue(
                    line_number=line_num,
                    severity=Severity.WARNING,
                    issue=f"Cannot validate {lib} compatibility: base CUDA version unknown",
                    recommendation="Use a CUDA base image with explicit version tag"
                )
                continue

            # Get DB-verified command for this CUDA and library
            verified_cmd = self.compat_db.get_verified_install_command(self.cuda_version, lib)

            if not verified_cmd:
                result.add_issue(
                    line_number=line_num,
                    severity=Severity.WARNING,
                    issue=f"No verified install command for {lib} on CUDA {self.cuda_version} in env-doctor DB",
                    recommendation=f"Verify {lib} compatibility with CUDA {self.cuda_version} manually or use a different CUDA version"
                )
                continue

            # If user pinned a specific version, check if it matches DB verified combo
            if pinned_version:
                pin_pattern = f"{lib}=={pinned_version}"

                if pin_pattern in verified_cmd:
                    # Pinned version matches DB - this is good, but keep quiet to avoid noise
                    pass
                else:
                    # Pinned version differs from DB verified combo
                    # Find other CUDA versions that support this pinned version
                    compatible_cudas = self.compat_db.find_cuda_versions_for_library_version(lib, pinned_version)

                    recommendation = f"Pinned {lib}=={pinned_version} is not verified in env-doctor's database for CUDA {self.cuda_version}. "

                    if compatible_cudas:
                        recommendation += f"This version is verified for CUDA: {', '.join(compatible_cudas)}. "

                    recommendation += f"Alternatively, use the DB-verified command for CUDA {self.cuda_version}."

                    result.add_issue(
                        line_number=line_num,
                        severity=Severity.WARNING,
                        issue=f"Pinned {lib}=={pinned_version} differs from DB-verified combination for CUDA {self.cuda_version}",
                        recommendation=recommendation,
                        corrected_command=f"RUN {verified_cmd}"
                    )

    def _validate_multi_library_compatibility(self, result: ValidationResult) -> None:
        """
        Emit a summary when multiple GPU libraries are installed, showing DB verification status.

        Args:
            result: ValidationResult to add issues to
        """
        if not self.cuda_version:
            return

        # Check for multiple GPU libraries
        gpu_libs = ["torch", "tensorflow", "jax"]
        detected = [lib for lib in gpu_libs if lib in self.detected_libraries]

        if len(detected) < 2:
            # Less than 2 GPU libs - no multi-lib summary needed
            return

        # Check verification status for each library
        verification_status = {}
        for lib in detected:
            verified_cmd = self.compat_db.get_verified_install_command(self.cuda_version, lib)
            verification_status[lib] = bool(verified_cmd)

        all_verified = all(verification_status.values())
        none_verified = not any(verification_status.values())

        # Find a reasonable line number (use the first detected lib's line)
        line_num = self.detected_libraries[detected[0]]["line_number"]

        if all_verified:
            # All libraries have DB-verified commands for this CUDA - good!
            libs_str = ", ".join(detected)
            result.add_issue(
                line_number=line_num,
                severity=Severity.INFO,
                issue=f"Multiple GPU libraries detected ({libs_str}) - all have verified combinations for CUDA {self.cuda_version}",
                recommendation=f"All detected libraries are verified in env-doctor's compatibility database for CUDA {self.cuda_version}."
            )
        else:
            # Some or none verified
            verified_libs = [lib for lib, verified in verification_status.items() if verified]
            unverified_libs = [lib for lib, verified in verification_status.items() if not verified]

            issue_parts = []
            if verified_libs:
                issue_parts.append(f"Verified: {', '.join(verified_libs)}")
            if unverified_libs:
                issue_parts.append(f"Not verified: {', '.join(unverified_libs)}")

            issue = f"Multiple GPU libraries detected for CUDA {self.cuda_version}. {'; '.join(issue_parts)}"

            # Try to find a CUDA version where all are verified
            all_cudas = self.compat_db.all_cuda_versions()
            compatible_cuda = None

            for cuda in sorted(all_cudas, reverse=True):
                if all(self.compat_db.get_verified_install_command(cuda, lib) for lib in detected):
                    compatible_cuda = cuda
                    break

            recommendation = f"Some libraries are not verified for CUDA {self.cuda_version}. "
            if compatible_cuda:
                recommendation += f"Consider using CUDA {compatible_cuda} where all detected libraries have verified combinations."
            else:
                recommendation += "Verify compatibility manually."

            result.add_issue(
                line_number=line_num,
                severity=Severity.WARNING,
                issue=issue,
                recommendation=recommendation
            )

    def _validate_cpu_base_with_gpu_libs(self, result: ValidationResult) -> None:
        """
        Check if CPU-only base image is used with GPU library installations.
        Provide DB-driven base image and install command recommendations.

        Args:
            result: ValidationResult to add issues to
        """
        if not self.base_image:
            return

        # Check if base is CPU-only
        if not is_cpu_only_image(self.base_image):
            return

        # Check if any GPU libraries are being installed
        gpu_libs = ["torch", "tensorflow", "jax"]
        detected_gpu_libs = [lib for lib in gpu_libs if lib in self.detected_libraries]

        if not detected_gpu_libs:
            # CPU base but no GPU libs - this is fine
            return

        # CPU-only base with GPU libraries detected - provide DB-driven recommendations
        # Prefer torch for recommendations if present
        primary_lib = "torch" if "torch" in detected_gpu_libs else detected_gpu_libs[0]
        lib_info = self.detected_libraries[primary_lib]
        pinned_version = lib_info["version"]

        # Build DB-driven recommendation
        recommended_cuda = None
        recommended_cmd = None

        if pinned_version:
            # Try to find a CUDA version that has this pinned version in DB
            compatible_cudas = self.compat_db.find_cuda_versions_for_library_version(primary_lib, pinned_version)
            if compatible_cudas:
                # Use the first compatible CUDA (could sort to prefer newer)
                recommended_cuda = compatible_cudas[0]
                recommended_cmd = self.compat_db.get_verified_install_command(recommended_cuda, primary_lib)

        if not recommended_cuda:
            # No pinned version or not found - use default CUDA with DB entry for this lib
            # Prefer 12.1 if it has an entry, otherwise use the first available
            all_cudas = self.compat_db.all_cuda_versions()
            if "12.1" in all_cudas and self.compat_db.get_verified_install_command("12.1", primary_lib):
                recommended_cuda = "12.1"
            else:
                # Find first CUDA with an entry for this lib
                for cuda in sorted(all_cudas, reverse=True):  # Prefer newer CUDA
                    if self.compat_db.get_verified_install_command(cuda, primary_lib):
                        recommended_cuda = cuda
                        break

            if recommended_cuda:
                recommended_cmd = self.compat_db.get_verified_install_command(recommended_cuda, primary_lib)

        # Build corrected command
        corrected = []
        if recommended_cuda:
            corrected.append(f"FROM nvidia/cuda:{recommended_cuda}.0-runtime-ubuntu22.04")
            if recommended_cmd:
                corrected.append(f"RUN {recommended_cmd}")
        else:
            # Fallback if no DB recommendation found
            corrected.append("FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04")
            corrected.append("# Add appropriate pip install commands with --index-url")

        # Find the FROM line number
        from_line = 1
        for line_num, line in self.lines:
            if line.upper().startswith('FROM '):
                from_line = line_num
                break

        result.add_issue(
            line_number=from_line,
            severity=Severity.ERROR,
            issue=f"CPU-only base image detected ({self.base_image}), but GPU libraries will be installed: {', '.join(detected_gpu_libs)}",
            recommendation=f"Use a GPU-enabled base image. For the detected libraries, consider CUDA {recommended_cuda or '12.1'}.",
            corrected_command="\n".join(corrected)
        )

    def _validate_runtime_devel_mismatch(self, result: ValidationResult) -> None:
        """
        Validate that compilation-required packages are not used with runtime-only base images.

        Args:
            result: ValidationResult to add issues to
        """
        if self.base_image_flavor != "runtime":
            # Not a runtime image, or flavor unknown - skip check
            return

        # Only the final stage determines what ships, so scope compilation
        # detection to it. A builder stage compiling in -devel is correct and
        # must not be reported against a -runtime final stage.
        needs_compilation, compilation_line = self._detect_compilation(self.final_stage_lines)

        if not needs_compilation:
            return

        # Runtime image with compilation needs - provide recommendation
        corrected_image = None
        if self.base_image and "nvidia/cuda" in self.base_image.lower():
            # Try to switch runtime to devel in the same image
            corrected_image = self.base_image.replace("-runtime", "-devel")

        recommendation = "CUDA compilation required but base image is -runtime (nvcc missing). Use a -devel base image instead."
        if corrected_image:
            recommendation += f" Alternatively, consider pre-built wheels or a multi-stage build to reduce final image size."

        # Report against the final stage's FROM
        from_line = self.stages[-1]["line_number"] if self.stages else 1

        result.add_issue(
            line_number=from_line,
            severity=Severity.ERROR,
            issue=f"CUDA compilation required (detected compilation keywords at line {compilation_line}) but base image is -runtime",
            recommendation=recommendation,
            corrected_command=f"FROM {corrected_image}" if corrected_image else None
        )

    def _validate_image_bloat(self, result: ValidationResult) -> None:
        """
        Flag -devel base images that carry the full toolkit without needing it.

        A -devel image ships nvcc, headers and static libraries. When nothing in
        the Dockerfile compiles CUDA code, that is several gigabytes of dead
        weight in the shipped image.

        This inverts the runtime/devel heuristic, so it is deliberately
        conservative: a missed compilation signal here would tell the user to
        change base images and break their build. Reported as INFO, and
        suppressed whenever dependencies are opaque.

        Args:
            result: ValidationResult to add issues to
        """
        if self.base_image_variant not in ("devel", "cudnn-devel"):
            return

        # Any compilation anywhere in the file justifies the devel image
        needs_compilation, _ = self._detect_compilation(self.lines)
        if needs_compilation:
            return

        # Cannot see what requirements.txt / a local project pulls in
        if self._has_opaque_dependencies(self.lines):
            return

        downgrade_paths = self.image_sizes.get("downgrade_paths", {})
        targets = downgrade_paths.get(self.base_image_variant, [])
        if not targets:
            return

        target = targets[0]
        savings = self._format_savings(self.base_image_variant, target)

        corrected = None
        if self.base_image and "-devel" in self.base_image.lower():
            corrected = f"FROM {self.base_image.replace('-devel', '-' + target.replace('cudnn-', ''))}"

        recommendation = (
            f"No CUDA compilation was detected, so the full toolkit is not needed at runtime. "
            f"A -{target.replace('cudnn-', '')} image provides the same CUDA runtime libraries"
        )
        if savings:
            recommendation += f" and is {savings}"
        recommendation += "."

        stage_note = " in the final stage" if len(self.stages) > 1 else ""

        result.add_issue(
            line_number=self.stages[-1]["line_number"] if self.stages else 1,
            severity=Severity.INFO,
            issue=f"Base image{stage_note} is -{self.base_image_variant} but no CUDA compilation was detected",
            recommendation=recommendation,
            corrected_command=corrected
        )

    def _suggest_multi_stage_build(self, result: ValidationResult) -> None:
        """
        Suggest a multi-stage build when compilation forces a -devel base image.

        Compiling needs the toolkit, but shipping it does not. Splitting the
        build lets the compiled artifacts land in a slim runtime image.

        Only fires for single-stage Dockerfiles - a file that already has a
        builder stage is doing the right thing.

        Args:
            result: ValidationResult to add issues to
        """
        if self.base_image_variant not in ("devel", "cudnn-devel"):
            return

        # Already multi-stage - nothing to suggest
        if len(self.stages) > 1:
            return

        needs_compilation, compilation_line = self._detect_compilation(self.final_stage_lines)
        if not needs_compilation:
            return

        downgrade_paths = self.image_sizes.get("downgrade_paths", {})
        targets = downgrade_paths.get(self.base_image_variant, [])
        target = targets[0] if targets else "runtime"
        savings = self._format_savings(self.base_image_variant, target)

        runtime_image = self.base_image
        if self.base_image and "-devel" in self.base_image.lower():
            runtime_image = self.base_image.replace(
                "-devel", "-" + target.replace("cudnn-", "")
            )

        skeleton = (
            f"FROM {self.base_image} AS builder\n"
            f"# ... compilation steps (line {compilation_line}) stay here ...\n"
            f"\n"
            f"FROM {runtime_image}\n"
            f"COPY --from=builder /usr/local/lib/python3/site-packages "
            f"/usr/local/lib/python3/site-packages"
        )

        recommendation = (
            "Compilation needs the toolkit but the shipped image does not. "
            "Compile in a -devel builder stage, then copy the built artifacts into a "
            f"-{target.replace('cudnn-', '')} final stage"
        )
        if savings:
            recommendation += f" ({savings})"
        recommendation += "."

        result.add_issue(
            line_number=self.stages[-1]["line_number"] if self.stages else 1,
            severity=Severity.INFO,
            issue=(
                f"Single-stage -{self.base_image_variant} image compiles CUDA code "
                f"(line {compilation_line}); the toolkit ships to production"
            ),
            recommendation=recommendation,
            corrected_command=skeleton
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
        Flag CUDA toolkit installations that are unnecessary or redundant.

        Two distinct cases:
        - Redundant: installing CUDA packages on top of an image that already
          provides them (nvidia/cuda and friends).
        - Unnecessary: installing the toolkit when nothing compiles CUDA code.

        Args:
            result: ValidationResult to add issues to
        """
        # Scope to the final stage: toolkit installed in a builder stage is
        # discarded and does not bloat the shipped image.
        stage_lines = self.final_stage_lines or self.lines

        needs_compilation, _ = self._detect_compilation(stage_lines)

        # A CUDA base image already ships the runtime libraries
        base_provides_cuda = bool(
            self.base_image and is_gpu_enabled_image(self.base_image)
        )

        toolkit_patterns = [
            r'apt-get\s+install.*cuda-toolkit',
            r'apt\s+install.*cuda-toolkit',
            r'apt-get\s+install.*\bcuda\b',
            r'apt\s+install.*\bcuda\b',
            r'yum\s+install.*\bcuda\b',
            r'dnf\s+install.*\bcuda\b',
            r'conda\s+install.*\bcudatoolkit\b',
            r'mamba\s+install.*\bcudatoolkit\b',
        ]

        for line_num, line in stage_lines:
            upper = line.upper()
            if not (upper.startswith('RUN') or upper.startswith('CONDA')):
                continue

            line_lower = line.lower()

            # Skip if it's just libcuda
            if 'libcuda' in line_lower and 'cuda-toolkit' not in line_lower:
                continue

            for pattern in toolkit_patterns:
                if not re.search(pattern, line_lower):
                    continue

                if base_provides_cuda and not needs_compilation:
                    result.add_issue(
                        line_number=line_num,
                        severity=Severity.WARNING,
                        issue=(
                            f"Redundant CUDA installation on top of a CUDA base image "
                            f"({self.base_image})"
                        ),
                        recommendation=(
                            "The base image already provides the CUDA runtime. This "
                            "installation duplicates it and adds 2-5GB. Remove it, or "
                            "switch to a -devel base image if you genuinely need the toolkit."
                        ),
                        corrected_command="# Remove - CUDA is already provided by the base image"
                    )
                elif needs_compilation:
                    result.add_issue(
                        line_number=line_num,
                        severity=Severity.INFO,
                        issue="CUDA toolkit installation detected",
                        recommendation="Toolkit appears needed for compilation (flash-attention, xformers, etc.). "
                                     "This adds 2-5GB to image size. Consider using pre-built wheels if available.",
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
