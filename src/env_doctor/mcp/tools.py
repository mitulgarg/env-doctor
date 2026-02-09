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