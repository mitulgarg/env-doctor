"""
MCP tool implementations for env-doctor.

These functions wrap existing env-doctor modules to provide
read-only diagnostic tools for AI assistants.
"""

from typing import Any, Dict, List, Optional
import tempfile
import os


def env_check() -> Dict[str, Any]:
    """
    Run full environment diagnostics using all registered detectors.

    Returns:
        Dict containing results from all detectors with keys:
        - components: Dict mapping component name to detection result
        - summary: Overall status summary
    """
    # Import detectors to trigger registration
    from env_doctor.detectors import nvidia_driver, cuda_toolkit, cudnn, python_libraries, wsl2
    from env_doctor.core.registry import DetectorRegistry

    results = {}
    overall_status = "success"

    for detector in DetectorRegistry.all():
        if not detector.can_run():
            continue

        try:
            result = detector.detect()
            results[result.component] = result.to_dict()

            # Update overall status
            if result.status.value == "error":
                overall_status = "error"
            elif result.status.value == "warning" and overall_status != "error":
                overall_status = "warning"
        except Exception as e:
            results[detector.name] = {
                "component": detector.name,
                "status": "error",
                "detected": False,
                "error": str(e),
            }
            overall_status = "error"

    return {
        "components": results,
        "summary": {
            "status": overall_status,
            "component_count": len(results),
            "detected_count": sum(1 for r in results.values() if r.get("detected", False)),
        },
    }


def env_check_component(component: str) -> Dict[str, Any]:
    """
    Run diagnostics for a specific component.

    Args:
        component: Component name (nvidia_driver, cuda_toolkit, cudnn, python_library, wsl2)

    Returns:
        Dict containing detection result for the specified component
    """
    # Import detectors to trigger registration
    from env_doctor.detectors import nvidia_driver, cuda_toolkit, cudnn, python_libraries, wsl2
    from env_doctor.core.registry import DetectorRegistry
    from env_doctor.core.exceptions import DetectorNotFoundError

    valid_components = DetectorRegistry.get_names()

    if component not in valid_components:
        return {
            "error": f"Unknown component: {component}",
            "valid_components": valid_components,
        }

    try:
        detector = DetectorRegistry.get(component)

        if not detector.can_run():
            return {
                "component": component,
                "status": "skipped",
                "reason": "Detector cannot run on this platform",
            }

        result = detector.detect()
        return result.to_dict()

    except DetectorNotFoundError:
        return {
            "error": f"Component not found: {component}",
            "valid_components": valid_components,
        }
    except Exception as e:
        return {
            "component": component,
            "status": "error",
            "detected": False,
            "error": str(e),
        }


def model_check(model_name: str, precision: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if an AI model fits on available GPU hardware.

    Args:
        model_name: Name of the model (e.g., "llama-3-8b", "meta-llama/Llama-2-7b-hf")
        precision: Optional precision level (fp32, fp16, bf16, int8, int4, fp8)

    Returns:
        Dict with model compatibility analysis including:
        - success: Whether model was found
        - model_name: Normalized model name
        - gpu_info: Available GPU information
        - vram_requirements: VRAM needed for each precision
        - compatibility: Which precisions fit on GPU
        - recommendations: Actionable recommendations
    """
    # Import detectors to trigger registration (needed by ModelChecker)
    from env_doctor.detectors import nvidia_driver
    from env_doctor.utilities.model_checker import ModelChecker

    try:
        checker = ModelChecker()
        result = checker.check_compatibility(model_name, precision)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def model_list() -> Dict[str, Any]:
    """
    List all available models from the database.

    Returns:
        Dict with:
        - models_by_category: Models grouped by category (llm, diffusion, audio, etc.)
        - stats: Database statistics
    """
    from env_doctor.utilities.vram_calculator import VRAMCalculator

    try:
        calc = VRAMCalculator()
        models = calc.list_all_models()
        stats = calc.get_database_stats()

        return {
            "models_by_category": models,
            "stats": stats,
        }
    except Exception as e:
        return {
            "error": str(e),
        }


def dockerfile_validate(content: str) -> Dict[str, Any]:
    """
    Validate Dockerfile content for GPU/CUDA configuration issues.

    Args:
        content: Dockerfile content as a string

    Returns:
        Dict with validation results including:
        - file_path: Path to temporary file used
        - success: Whether validation passed (no errors)
        - error_count: Number of errors
        - warning_count: Number of warnings
        - info_count: Number of info messages
        - issues: List of validation issues with line numbers and recommendations
    """
    from env_doctor.validators.dockerfile_validator import DockerfileValidator

    # Write content to a temporary file for the validator
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dockerfile", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            validator = DockerfileValidator(dockerfile_path=temp_path)
            result = validator.validate()

            # Convert to dict
            issues = []
            for issue in result.issues:
                issues.append({
                    "line_number": issue.line_number,
                    "severity": issue.severity.value,
                    "issue": issue.issue,
                    "recommendation": issue.recommendation,
                    "corrected_command": issue.corrected_command,
                })

            return {
                "file_path": "<provided content>",
                "success": result.success,
                "error_count": result.error_count,
                "warning_count": result.warning_count,
                "info_count": result.info_count,
                "issues": issues,
            }
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def install_command(library: str) -> Dict[str, Any]:
    """
    Get safe installation command for a library based on detected GPU driver.

    Args:
        library: Library name (e.g., "torch", "tensorflow", "jax")

    Returns:
        Dict with:
        - library: The library name
        - driver_detected: Whether GPU driver was found
        - driver_version: Detected driver version (if found)
        - max_cuda: Maximum CUDA version supported by driver
        - install_command: The recommended pip install command
    """
    from env_doctor.detectors import nvidia_driver
    from env_doctor.core.registry import DetectorRegistry
    from env_doctor.db import get_max_cuda_for_driver, get_install_command

    try:
        driver_detector = DetectorRegistry.get("nvidia_driver")
        driver_result = driver_detector.detect()

        if not driver_result.detected:
            return {
                "library": library,
                "driver_detected": False,
                "install_command": f"pip install {library}",
                "note": "No NVIDIA driver found. Assuming CPU-only installation.",
            }

        max_cuda = driver_result.metadata.get("max_cuda_version", "Unknown")
        command = get_install_command(library, max_cuda)

        return {
            "library": library,
            "driver_detected": True,
            "driver_version": driver_result.version,
            "max_cuda": max_cuda,
            "install_command": command,
        }
    except Exception as e:
        return {
            "library": library,
            "error": str(e),
        }


def cuda_info() -> Dict[str, Any]:
    """
    Get detailed CUDA toolkit information.

    Returns:
        Dict with comprehensive CUDA toolkit analysis including:
        - detected: Whether CUDA toolkit was found
        - version: CUDA version
        - installations: List of all CUDA installations
        - nvcc: nvcc compiler info
        - cuda_home: CUDA_HOME environment variable status
        - libcudart: Runtime library info
        - path_config: PATH configuration status
        - ld_library_path: LD_LIBRARY_PATH status (Linux only)
        - driver_compatibility: Compatibility with installed driver
        - issues: List of detected issues
        - recommendations: List of recommendations
    """
    from env_doctor.detectors import cuda_toolkit
    from env_doctor.core.registry import DetectorRegistry

    try:
        detector = DetectorRegistry.get("cuda_toolkit")
        result = detector.detect()
        return result.to_dict()
    except Exception as e:
        return {
            "detected": False,
            "error": str(e),
        }


def cudnn_info() -> Dict[str, Any]:
    """
    Get detailed cuDNN library information.

    Returns:
        Dict with comprehensive cuDNN analysis including:
        - detected: Whether cuDNN was found
        - version: cuDNN version
        - path: Primary library path
        - libraries: List of all cuDNN libraries found
        - symlink_status: Symlink status (Linux)
        - path_status: PATH configuration (Windows)
        - cuda_compatibility: Compatibility with installed CUDA
        - issues: List of detected issues
        - recommendations: List of recommendations
    """
    from env_doctor.detectors import cudnn
    from env_doctor.core.registry import DetectorRegistry

    try:
        detector = DetectorRegistry.get("cudnn")

        if not detector.can_run():
            return {
                "detected": False,
                "status": "skipped",
                "reason": "cuDNN detector not supported on this platform",
            }

        result = detector.detect()
        return result.to_dict()
    except Exception as e:
        return {
            "detected": False,
            "error": str(e),
        }


def cuda_install(version: Optional[str] = None) -> Dict[str, Any]:
    """
    Get step-by-step CUDA Toolkit installation instructions.

    Args:
        version: Optional specific CUDA version to install (auto-detects from driver if not specified)

    Returns:
        Dict with:
        - platform: Detected platform information
        - recommended_version: Recommended CUDA Toolkit version
        - driver_version: Detected GPU driver version (if auto-detecting)
        - max_cuda: Maximum CUDA supported by driver
        - install_info: Platform-specific installation steps
        - download_page: Official download URL
    """
    from env_doctor.utilities.platform_detect import detect_platform
    from env_doctor.db import get_recommended_cuda_toolkit, get_cuda_install_steps, CUDA_INSTALL_DATA
    from env_doctor.detectors import nvidia_driver
    from env_doctor.core.registry import DetectorRegistry

    try:
        # Detect platform
        plat = detect_platform()
        result = {
            "platform": {
                "os": plat["os"],
                "distro": plat["distro"],
                "distro_version": plat["distro_version"],
                "arch": plat["arch"],
                "is_wsl2": plat["is_wsl2"],
                "platform_keys": plat["platform_keys"],
            }
        }

        # Determine CUDA version
        if version:
            recommended = version
            result["requested_version"] = version
        else:
            driver_detector = DetectorRegistry.get("nvidia_driver")
            driver_result = driver_detector.detect()

            if not driver_result.detected:
                return {
                    **result,
                    "error": "No NVIDIA driver detected. Install driver first.",
                    "driver_download": "https://www.nvidia.com/Download/index.aspx",
                }

            max_cuda = driver_result.metadata.get("max_cuda_version", "Unknown")
            result["driver_version"] = driver_result.version
            result["max_cuda"] = max_cuda

            recommended = get_recommended_cuda_toolkit(max_cuda)
            if not recommended:
                return {
                    **result,
                    "error": f"No installation instructions available for CUDA {max_cuda}",
                    "download_page": "https://developer.nvidia.com/cuda-downloads",
                }

        result["recommended_version"] = recommended

        # Get installation steps
        install_info = get_cuda_install_steps(recommended, plat["platform_keys"])

        if not install_info:
            # Get available platforms for this version
            versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
            version_data = versions.get(recommended, {})
            available = version_data.get("platforms", {})

            return {
                **result,
                "error": "No specific instructions for your platform",
                "available_platforms": [
                    {"key": k, "label": v.get("label", k)}
                    for k, v in available.items()
                ],
                "download_page": "https://developer.nvidia.com/cuda-downloads",
            }

        result["install_info"] = install_info
        return result

    except Exception as e:
        return {
            "error": str(e),
        }


def docker_compose_validate(content: str) -> Dict[str, Any]:
    """
    Validate docker-compose.yml content for GPU configuration issues.

    Args:
        content: docker-compose.yml content as a string

    Returns:
        Dict with validation results including:
        - success: Whether validation passed (no errors)
        - error_count: Number of errors
        - warning_count: Number of warnings
        - info_count: Number of info messages
        - issues: List of validation issues with recommendations
    """
    from env_doctor.validators.compose_validator import ComposeValidator

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            validator = ComposeValidator(compose_path=temp_path)
            result = validator.validate()

            # Convert to dict
            issues = []
            for issue in result.issues:
                issues.append({
                    "line_number": issue.line_number,
                    "severity": issue.severity.value,
                    "issue": issue.issue,
                    "recommendation": issue.recommendation,
                    "corrected_command": issue.corrected_command,
                })

            return {
                "file_path": "<provided content>",
                "success": result.success,
                "error_count": result.error_count,
                "warning_count": result.warning_count,
                "info_count": result.info_count,
                "issues": issues,
            }
        finally:
            os.unlink(temp_path)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }