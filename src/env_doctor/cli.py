"""
Modernized CLI using the detector architecture.

This version demonstrates how to migrate from procedural checks
to the new detector-based system.
"""
import sys
import os
import argparse
import platform
import json
from datetime import datetime
from typing import Dict, Any, Optional
from .core.registry import DetectorRegistry
from .core.detector import Status
from .db import get_max_cuda_for_driver, get_install_command, DB_DATA, get_recommended_cuda_toolkit, get_cuda_install_steps, CUDA_INSTALL_DATA

# Legacy imports for functions not yet refactored
from .checks import (
    scan_imports_in_folder,
    check_broken_imports
)

# Import detectors to register them!
# This triggers all @register decorators
from .detectors.nvidia_driver import NvidiaDriverDetector
from .detectors.cuda_toolkit import CudaToolkitDetector
from .detectors.python_libraries import PythonLibraryDetector
from .detectors.wsl2 import WSL2Detector
from .detectors.cudnn import CudnnDetector
from .detectors.python_compat import PythonCompatDetector

# Import model checker for model compatibility
from .utilities import ModelChecker

# Compilation packages that require nvcc CUDA version to match PyTorch's bundled CUDA
COMPILATION_PACKAGES = {
    "flash-attn": ["flash-attn", "flash-attention", "flash_attn"],
    "auto-gptq": ["auto-gptq", "autogptq", "auto_gptq"],
    "apex": ["apex"],
    "xformers": ["xformers"],
    "sageattention": ["sageattention", "sage-attention", "sage_attention"]
}

def check_compilation_health(cuda_result, torch_result):
    """
    Check if system CUDA matches PyTorch CUDA for compilation compatibility.
    
    Args:
        cuda_result: DetectionResult from CudaToolkitDetector
        torch_result: DetectionResult from PythonLibraryDetector (torch)
    """
    print("\n🏭  COMPILATION HEALTH (For Flash-Attention/AutoGPTQ)")
    
    if not cuda_result.detected:
        print("⚠️   System CUDA (nvcc) NOT found.")
        print("    -> You cannot install 'flash-attention' or 'auto-gptq' from source.")
        for rec in cuda_result.recommendations:
            print(f"    → {rec}")
        return
    
    if not torch_result.detected:
        if torch_result.status == Status.ERROR:
            print("❌  PyTorch import failed. Cannot check compilation compatibility.")
            for issue in torch_result.issues:
                print(f"    {issue}")
        else:
            print("❓  PyTorch not installed. Skipping check.")
        return
    
    torch_cuda = torch_result.metadata.get("cuda_version", "Unknown")
    if torch_cuda == "Unknown":
        print("❓  Torch CUDA version unknown. Skipping check.")
        return

    sys_cuda = cuda_result.version
    sys_mm = ".".join(sys_cuda.split(".")[:2])
    torch_mm = ".".join(torch_cuda.split(".")[:2])

    if sys_mm == torch_mm:
        print(f"✅  PERFECT SYMMETRY: System ({sys_cuda}) == Torch ({torch_cuda})")
    else:
        print(f"❌  ASYMMETRY DETECTED: System ({sys_cuda}) != Torch ({torch_cuda})")
        print("    -> pip install flash-attn/auto-gptq/apex/xformers will likely FAIL.")
        print(f"    -> Run 'env-doctor install <package>' to see fix options")
        print(f"    -> Or manually install CUDA Toolkit {torch_mm}")


def check_compute_capability_compatibility(driver_result, torch_result):
    """
    Check if the GPU's compute capability is supported by the installed PyTorch.

    When new GPUs release, stable PyTorch wheels may lack compiled kernels for the
    latest SM architecture. This causes torch.cuda.is_available() to return False
    even though driver and CUDA toolkit work fine.

    Args:
        driver_result: DetectionResult from NvidiaDriverDetector
        torch_result: DetectionResult from PythonLibraryDetector (torch)

    Returns:
        dict with compatibility info for JSON output, or None
    """
    from .detectors.compute_capability import (
        get_sm_for_compute_capability,
        get_arch_name,
        is_sm_in_arch_list,
    )

    gpu_cc = driver_result.metadata.get("primary_gpu_compute_capability")
    arch_list = torch_result.metadata.get("arch_list", [])
    gpu_name = driver_result.metadata.get("primary_gpu_name", "Unknown GPU")
    torch_version = torch_result.version or "Unknown"
    torch_cuda = torch_result.metadata.get("cuda_version", "Unknown")

    # Build result dict for JSON output
    compat_info = {
        "gpu_name": gpu_name,
        "compute_capability": gpu_cc,
        "arch_list": arch_list,
    }

    if not gpu_cc:
        compat_info["status"] = "unknown"
        compat_info["message"] = "Could not determine GPU compute capability"
        return compat_info

    sm = get_sm_for_compute_capability(gpu_cc)
    arch_name = get_arch_name(gpu_cc)

    compat_info["sm"] = sm
    compat_info["arch_name"] = arch_name

    if not arch_list:
        compat_info["status"] = "unknown"
        compat_info["message"] = "Could not determine PyTorch arch list"
        return compat_info

    if is_sm_in_arch_list(sm, arch_list):
        compat_info["status"] = "compatible"
        compat_info["message"] = f"PyTorch supports {sm} ({arch_name})"
        print(f"\n\U0001f3af  COMPUTE CAPABILITY CHECK")
        print(f"    GPU: {gpu_name} (Compute {gpu_cc}, {arch_name}, {sm})")
        print(f"    PyTorch compiled for: {', '.join(arch_list)}")
        print(f"    \u2705 COMPATIBLE: PyTorch {torch_version} supports your GPU architecture.")
    else:
        compat_info["status"] = "mismatch"
        compat_info["message"] = f"PyTorch {torch_version} does not support {sm}"

        # Build nightly URL
        cuda_slug = ""
        if torch_cuda and torch_cuda != "Unknown":
            cuda_slug = "cu" + torch_cuda.replace(".", "").split()[0]
        nightly_url = f"https://download.pytorch.org/whl/nightly/{cuda_slug}" if cuda_slug else "https://download.pytorch.org/whl/nightly/"
        compat_info["nightly_url"] = nightly_url

        print(f"\n\U0001f3af  COMPUTE CAPABILITY CHECK")
        print(f"    GPU: {gpu_name} (Compute {gpu_cc}, {arch_name}, {sm})")
        print(f"    PyTorch compiled for: {', '.join(arch_list)}")
        print(f"    \u274c ARCHITECTURE MISMATCH: Your GPU needs {sm} but PyTorch {torch_version} doesn't include it.")
        print()
        print(f"    This is why torch.cuda.is_available() returns False even though")
        print(f"    your driver and CUDA toolkit are working correctly.")
        print()
        print(f"    FIX: Install PyTorch nightly with {sm} support:")
        print(f"       pip install --pre torch torchvision torchaudio --index-url {nightly_url}")

    return compat_info


def check_system_path():
    """Check LD_LIBRARY_PATH for CUDA linking (TensorFlow/JAX)."""
    print("\n🔗  SYSTEM LINKING (For TensorFlow/JAX)")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        print("⚠️   LD_LIBRARY_PATH is unset.")
        print("    → For TensorFlow/JAX, you may need to set this.")
        return
    print(f"    LD_LIBRARY_PATH: {ld_path}")
    if "cuda" not in ld_path.lower():
        print("⚠️   Warning: LD_LIBRARY_PATH is set but does not seem to point to any CUDA folders.")


def print_detection_result(result, emoji="📦"):
    """
    Pretty-print a DetectionResult.
    
    Args:
        result: DetectionResult object
        emoji: Emoji prefix for the component
    """
    component_name = result.component.replace("_", " ").title()
    
    if result.status == Status.SUCCESS:
        print(f"✅  {component_name}: {result.version}")
        if result.path:
            print(f"    Path: {result.path}")
        
        # Print metadata
        for key, value in result.metadata.items():
            if key not in ["detection_method"]:  # Skip internal keys
                display_key = key.replace("_", " ").title()
                print(f"    → {display_key}: {value}")
    
    elif result.status == Status.NOT_FOUND:
        print(f"❌  {component_name}: Not Found")
    
    elif result.status == Status.WARNING:
        print(f"⚠️   {component_name}: {result.version or 'Warning'}")
    
    elif result.status == Status.ERROR:
        print(f"❌  {component_name}: Error")
    
    # Print issues
    for issue in result.issues:
        print(f"    ⚠️  {issue}")
    
    # Print recommendations
    for rec in result.recommendations:
        print(f"    → {rec}")


def check_library_compatibility(lib_result, max_cuda):
    """
    Check if a library's CUDA version is compatible with the driver.

    Args:
        lib_result: DetectionResult from PythonLibraryDetector
        max_cuda: Maximum CUDA version supported by driver (string)
    """
    if not lib_result.detected or not max_cuda:
        return

    lib_cuda = lib_result.metadata.get("cuda_version", "Unknown")
    if lib_cuda == "Unknown" or "CPU" in lib_cuda:
        return

    try:
        # Extract numeric CUDA version
        cuda_num = lib_cuda.split(" ")[0].replace("x", "0")
        if float(cuda_num) > float(max_cuda):
            lib_name = lib_result.component.replace("python_library_", "")
            print(f"    ❌ CRITICAL CONFLICT: {lib_name} uses CUDA {lib_cuda}, Driver supports up to {max_cuda}!")
            print(f"    → Run 'env-doctor install {lib_name}' to fix.")
        else:
            print(f"    ✅ Compatible with Driver (CUDA {max_cuda})")
    except (ValueError, IndexError):
        pass


def determine_overall_status(results: Dict[str, Any]) -> str:
    """
    Determine overall status from detection results.

    Args:
        results: Dictionary of detection results

    Returns:
        str: "pass", "warning", or "fail"
    """
    has_errors = False
    has_warnings = False

    # Check each result
    for key, result in results.items():
        if result is None:
            continue

        if isinstance(result, dict):
            # Handle libraries dict
            for lib_result in result.values():
                if lib_result.status == Status.ERROR:
                    has_errors = True
                elif lib_result.status in [Status.WARNING, Status.NOT_FOUND]:
                    has_warnings = True
        else:
            # Handle single result
            if result.status == Status.ERROR:
                has_errors = True
            elif result.status in [Status.WARNING, Status.NOT_FOUND]:
                has_warnings = True

    if has_errors:
        return "fail"
    elif has_warnings:
        return "warning"
    else:
        return "pass"


def count_issues(results: Dict[str, Any]) -> int:
    """
    Count total issues across all detection results.

    Args:
        results: Dictionary of detection results

    Returns:
        int: Total number of issues
    """
    count = 0

    for key, result in results.items():
        if result is None:
            continue

        if isinstance(result, dict):
            # Handle libraries dict
            for lib_result in result.values():
                count += len(lib_result.issues)
        else:
            # Handle single result
            count += len(result.issues)

    return count


def determine_exit_code(results: Dict[str, Any]) -> int:
    """
    Determine exit code based on detection results.

    Args:
        results: Dictionary of detection results

    Returns:
        int: Exit code (0 = pass, 1 = warnings/failures, 2 = errors)
    """
    has_errors = False
    has_warnings = False

    for key, result in results.items():
        if result is None:
            continue

        if isinstance(result, dict):
            # Handle libraries dict
            for lib_result in result.values():
                if lib_result.status == Status.ERROR:
                    has_errors = True
                elif lib_result.status in [Status.WARNING, Status.NOT_FOUND]:
                    has_warnings = True
        else:
            # Handle single result
            if result.status == Status.ERROR:
                has_errors = True
            elif result.status in [Status.WARNING, Status.NOT_FOUND]:
                has_warnings = True

    if has_errors:
        return 2
    elif has_warnings:
        return 1
    else:
        return 0


def check_command(output_json: bool = False, ci: bool = False):
    """
    Main diagnostic command using detector architecture.

    This is the MODERNIZED version that uses DetectorRegistry
    instead of direct function calls.

    Args:
        output_json: Output as JSON (machine-readable)
        ci: CI-friendly mode (implies JSON + proper exit codes)
    """
    # === Collect all detection results ===
    # STEP 1: Environment Detection
    wsl2_detector = DetectorRegistry.get("wsl2")
    wsl2_result = wsl2_detector.detect() if wsl2_detector.can_run() else None

    # STEP 2: Hardware Detection
    driver_detector = DetectorRegistry.get("nvidia_driver")
    driver_result = driver_detector.detect()
    max_cuda = driver_result.metadata.get("max_cuda_version", None) if driver_result.detected else None

    # STEP 3: System CUDA Detection
    cuda_detector = DetectorRegistry.get("cuda_toolkit")
    cuda_result = cuda_detector.detect()

    # STEP 4: cuDNN Detection
    cudnn_detector = DetectorRegistry.get("cudnn")
    cudnn_result = cudnn_detector.detect() if cudnn_detector.can_run() else None

    # STEP 5: Python Libraries Detection
    libs = ["torch", "tensorflow", "jax"]
    torch_result = None
    lib_results = {}

    from .detectors.python_libraries import PythonLibraryDetector

    for lib in libs:
        lib_detector = PythonLibraryDetector(lib)
        lib_result = lib_detector.detect()
        lib_results[lib] = lib_result

        if lib == "torch":
            torch_result = lib_result

    # STEP 6: Python Compatibility Check
    python_compat_detector = DetectorRegistry.get("python_compat")
    python_compat_result = python_compat_detector.detect()

    # Organize results for JSON output
    results = {
        "wsl2": wsl2_result,
        "driver": driver_result,
        "cuda": cuda_result,
        "cudnn": cudnn_result,
        "libraries": lib_results,
        "python_compat": python_compat_result,
    }

    # === Choose output format ===
    # Compute capability check (for both JSON and human output)
    compute_compat_info = None
    if torch_result and torch_result.detected and driver_result.detected:
        # For JSON mode, we compute silently; for human mode, it prints inline
        if ci or output_json:
            from .detectors.compute_capability import (
                get_sm_for_compute_capability,
                get_arch_name,
                is_sm_in_arch_list,
            )
            gpu_cc = driver_result.metadata.get("primary_gpu_compute_capability")
            arch_list = torch_result.metadata.get("arch_list", [])
            gpu_name = driver_result.metadata.get("primary_gpu_name", "Unknown GPU")
            torch_cuda = torch_result.metadata.get("cuda_version", "Unknown")

            compute_compat_info = {
                "gpu_name": gpu_name,
                "compute_capability": gpu_cc,
                "arch_list": arch_list,
            }

            if gpu_cc and arch_list:
                sm = get_sm_for_compute_capability(gpu_cc)
                arch_name_val = get_arch_name(gpu_cc)
                compute_compat_info["sm"] = sm
                compute_compat_info["arch_name"] = arch_name_val
                compatible = is_sm_in_arch_list(sm, arch_list)
                compute_compat_info["status"] = "compatible" if compatible else "mismatch"
                if not compatible and torch_cuda and torch_cuda != "Unknown":
                    cuda_slug = "cu" + torch_cuda.replace(".", "").split()[0]
                    compute_compat_info["nightly_url"] = f"https://download.pytorch.org/whl/nightly/{cuda_slug}"
            else:
                compute_compat_info["status"] = "unknown"

    if ci or output_json:
        # JSON output
        output = {
            "status": determine_overall_status(results),
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "driver": "found" if driver_result.detected else "not_found",
                "cuda": "found" if cuda_result.detected else "not_found",
                "cudnn": "found" if (cudnn_result and cudnn_result.detected) else "not_found",
                "issues_count": count_issues(results)
            },
            "checks": {
                "wsl2": wsl2_result.to_dict() if wsl2_result else None,
                "driver": driver_result.to_dict(),
                "cuda": cuda_result.to_dict(),
                "cudnn": cudnn_result.to_dict() if cudnn_result else None,
                "libraries": {
                    lib: result.to_dict()
                    for lib, result in lib_results.items()
                },
                "python_compat": python_compat_result.to_dict(),
                "compute_compatibility": compute_compat_info,
            }
        }
        print(json.dumps(output, indent=2))
        sys.exit(determine_exit_code(results))
    else:
        # Human output (existing code)
        print("\n🩺  ENV-DOCTOR DIAGNOSIS  🩺")
        print("==============================")

        # --- Show DB Status ---
        meta = DB_DATA.get("_metadata", {})
        if meta:
            print(f"🛡️  DB Verified: {meta.get('last_verified', 'Unknown')}")
            print(f"    Method: {meta.get('method', 'Unknown')}")
            print("------------------------------")

        # === STEP 1: Environment Detection ===
        if wsl2_result:
            print_detection_result(wsl2_result, "🐧")
            print("------------------------------")

        # === STEP 2: Hardware Detection ===
        if driver_result.detected:
            print(f"✅  GPU Driver Found: {driver_result.version}")
            print(f"    → Max Supported CUDA: {max_cuda}")
            print(f"    → Detection Method: {driver_result.metadata.get('detection_method', 'unknown')}")
        else:
            print("⚠️   No NVIDIA Driver detected.")
            for rec in driver_result.recommendations:
                print(f"    → {rec}")

        # === STEP 3: System CUDA Detection ===
        if cuda_result.detected:
            print(f"✅  System CUDA (nvcc): {cuda_result.version}")
            if cuda_result.path:
                print(f"    Path: {cuda_result.path}")

            # Show quick status
            install_count = cuda_result.metadata.get("installation_count", 1)
            if install_count > 1:
                print(f"    ⚠️  {install_count} CUDA installations detected")

            if cuda_result.status == Status.WARNING:
                print(f"    ⚠️  Configuration issues detected (run 'doctor debug' for details)")
            elif cuda_result.status == Status.ERROR:
                print(f"    ❌ Critical issues detected (run 'doctor debug' for details)")
        else:
            print("ℹ️   System CUDA (nvcc) not found.")
            if cuda_result.recommendations:
                print(f"    → {cuda_result.recommendations[0]}")

        print("------------------------------")

        # cuDNN Detection
        if cudnn_result and cudnn_result.detected:
            print(f"✅  cuDNN: v{cudnn_result.version}")

        # === STEP 4: Python Libraries Detection ===
        for lib, lib_result in lib_results.items():
            if lib_result.detected:
                print(f"📦  Found {lib}: v{lib_result.version}")

                # Show bundled CUDA info
                cuda_ver = lib_result.metadata.get("cuda_version", "Unknown")
                if cuda_ver != "Unknown":
                    print(f"    → Bundled CUDA: {cuda_ver}")

                    # Check compatibility with driver
                    if max_cuda:
                        check_library_compatibility(lib_result, max_cuda)
                else:
                    print(f"    → Bundled CUDA: Not Detected")
            else:
                print(f"❌  {lib} is NOT installed.")

        # === STEP 5: Python Compatibility Check ===
        print("------------------------------")
        conflicts = python_compat_result.metadata.get("conflicts", [])
        if python_compat_result.status == Status.SUCCESS:
            checked = python_compat_result.metadata.get("constraints_checked", 0)
            print(f"✅  Python {python_compat_result.version}: Compatible with all {checked} checked libraries")
        elif python_compat_result.status == Status.ERROR:
            print(f"❌  Python {python_compat_result.version}: {len(conflicts)} compatibility issue(s)")
            for conflict in conflicts:
                print(f"    ⚠️  {conflict['message']}")
            for rec in python_compat_result.recommendations:
                print(f"    → {rec}")

        # === STEP 6: Compilation Health Check ===
        if torch_result and torch_result.detected:
            check_compilation_health(cuda_result, torch_result)

        # === STEP 6b: Compute Capability Check ===
        if torch_result and torch_result.detected and driver_result.detected:
            check_compute_capability_compatibility(driver_result, torch_result)

        # === STEP 7: System Path Check ===
        check_system_path()

        # === STEP 7: Code Migration Check ===
        # (Not yet refactored - still using legacy function)
        check_broken_imports()

        # === STEP 8: Offer detailed analysis ===
        if cuda_result.detected and (cuda_result.issues or cuda_result.metadata.get("installation_count", 1) > 1):
            print("\n💡  TIP: Run 'env-doctor cuda-info' for detailed CUDA analysis")



def python_compat_command(output_json: bool = False):
    """
    Check Python version compatibility with installed AI libraries.

    Args:
        output_json: Output as JSON (machine-readable)
    """
    detector = DetectorRegistry.get("python_compat")
    result = detector.detect()

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.status == Status.SUCCESS else 1)
    else:
        print("\n🐍  PYTHON VERSION COMPATIBILITY CHECK")
        print("=" * 60)
        print(f"Python Version: {result.version} ({result.metadata.get('python_full_version', '')})")
        print(f"Libraries Checked: {result.metadata.get('constraints_checked', 0)}")

        conflicts = result.metadata.get("conflicts", [])
        if not conflicts:
            print("\n✅  No compatibility issues found!")
        else:
            print(f"\n❌  {len(conflicts)} compatibility issue(s) found:")
            for conflict in conflicts:
                print(f"\n    {conflict['library']}:")
                print(f"      {conflict['message']}")
                if conflict.get("notes"):
                    print(f"      Note: {conflict['notes']}")

        cascades = result.metadata.get("cascades", [])
        if cascades:
            print("\n⚠️   Dependency Cascades:")
            for cascade in cascades:
                affected = ", ".join(cascade["affected_dependencies"])
                print(f"    {cascade['root_library']} [{cascade['severity']}]: {cascade['description']}")
                print(f"      Affected: {affected}")

        for rec in result.recommendations:
            print(f"\n💡  {rec}")

        print("\n" + "=" * 60)


def install_command(package_name):
    """
    Provide installation prescription for a package.

    Uses NvidiaDriverDetector to determine compatible CUDA version.
    For compilation packages (flash-attn, auto-gptq, apex, xformers),
    provides special two-option guidance.
    """
    # Check if this is a compilation package that requires nvcc/PyTorch CUDA matching
    for canonical_name, aliases in COMPILATION_PACKAGES.items():
        if package_name.lower() in aliases:
            compilation_package_install_prescription(canonical_name, package_name)
            return

    print(f"\n🩺  PRESCRIPTION FOR: {package_name}")

    # Use detector instead of direct function call
    driver_detector = DetectorRegistry.get("nvidia_driver")
    driver_result = driver_detector.detect()
    
    if not driver_result.detected:
        print("⚠️  No NVIDIA Driver found. Assuming CPU-only.")
        print(f"   pip install {package_name}")
        return

    max_cuda = driver_result.metadata.get("max_cuda_version", "Unknown")
    print(f"Detected Driver: {driver_result.version} (Supports up to CUDA {max_cuda})")
    
    command = get_install_command(package_name, max_cuda)
    print("\n⬇️   Run this command:")
    print("---------------------------------------------------")
    print(command)
    print("---------------------------------------------------")


def compilation_package_install_prescription(canonical_name, package_input):
    """
    Provide installation prescription for packages requiring nvcc/PyTorch CUDA match.

    Packages like flash-attn, auto-gptq, apex, and xformers need to be compiled from
    source and require exact CUDA version match between system nvcc and PyTorch's
    bundled CUDA.

    Args:
        canonical_name: Canonical package name (e.g., "flash-attn")
        package_input: User's input (e.g., "flash-attention" or "flash_attn")
    """
    print(f"\n🩺  PRESCRIPTION FOR: {canonical_name}")

    # Detect current state using existing detectors
    cuda_detector = DetectorRegistry.get("cuda_toolkit")
    cuda_result = cuda_detector.detect()

    torch_detector = PythonLibraryDetector("torch")
    torch_result = torch_detector.detect()

    # Handle edge case: No nvcc found
    if not cuda_result.detected:
        _show_no_nvcc_guidance(canonical_name, torch_result)
        return

    # Handle edge case: No PyTorch found or import failed
    if not torch_result.detected:
        if torch_result.status == Status.ERROR:
            # PyTorch is installed but failed to import
            print(f"\n❌  PyTorch import failed!")
            for issue in torch_result.issues:
                print(f"    {issue}")
            print("\n💡  Recommendations:")
            for rec in torch_result.recommendations:
                print(f"    • {rec}")
        else:
            # PyTorch not installed
            _show_no_pytorch_guidance(canonical_name)
        return

    # Extract CUDA versions
    torch_cuda = torch_result.metadata.get("cuda_version", "Unknown")
    if torch_cuda == "Unknown":
        print("\n⚠️  Cannot determine PyTorch's CUDA version.")
        print(f"    Install {canonical_name} at your own risk:")
        print(f"    pip install {canonical_name} --no-build-isolation")
        return

    sys_cuda = cuda_result.version
    sys_mm = ".".join(sys_cuda.split(".")[:2])  # Major.minor
    torch_mm = ".".join(torch_cuda.split(".")[:2])  # Major.minor

    # Handle edge case: Versions already match
    if sys_mm == torch_mm:
        _show_perfect_match_guidance(canonical_name, sys_cuda, torch_cuda)
        return

    # Main case: CUDA version mismatch - show two options
    _show_prescription_header(canonical_name, sys_cuda, torch_cuda)
    _show_option1_pytorch_downgrade(canonical_name, sys_mm, sys_cuda)
    _show_option2_nvcc_upgrade(canonical_name, torch_mm, torch_cuda, cuda_result)


def _show_no_nvcc_guidance(canonical_name, torch_result):
    """Show guidance when nvcc is not found."""
    print(f"\n❌  No nvcc found. You MUST install CUDA Toolkit to compile {canonical_name}")

    if torch_result.detected:
        torch_cuda = torch_result.metadata.get("cuda_version", "Unknown")
        if torch_cuda != "Unknown":
            torch_mm = ".".join(torch_cuda.split(".")[:2])
            print(f"\n📥  RECOMMENDED ACTION:")
            print(f"    1. Install CUDA Toolkit {torch_mm}")
            print(f"       https://developer.nvidia.com/cuda-{torch_mm.replace('.', '-')}-0-download-archive")
            print("    2. Verify: nvcc --version")
            print(f"    3. Install: pip install {canonical_name} --no-build-isolation")
        else:
            print(f"\n    Install CUDA Toolkit and then: pip install {canonical_name} --no-build-isolation")
    else:
        print(f"\n    Install CUDA Toolkit and PyTorch first.")


def _show_no_pytorch_guidance(canonical_name):
    """Show guidance when PyTorch is not found."""
    print(f"\n❌  PyTorch not installed. Install PyTorch first.")
    print("    Run: env-doctor install torch")
    print(f"\n    Then install {canonical_name} after PyTorch is ready.")


def _show_perfect_match_guidance(canonical_name, sys_cuda, torch_cuda):
    """Show success path when nvcc and PyTorch CUDA versions match."""
    print(f"\n✅  PERFECT MATCH: nvcc ({sys_cuda}) == PyTorch ({torch_cuda})")
    print(f"\n🎯  You can install {canonical_name} directly:")
    print(f"    pip install {canonical_name} --no-build-isolation")


def _show_prescription_header(canonical_name, sys_cuda, torch_cuda):
    """Show the mismatch warning header."""
    print(f"\n⚠️   CUDA VERSION MISMATCH DETECTED")
    print(f"     System nvcc: {sys_cuda}")
    print(f"     PyTorch CUDA: {torch_cuda}")
    print(f"\n🔧  {canonical_name} requires EXACT CUDA version match for compilation.")
    print("    You have TWO options to fix this:")


def _show_option1_pytorch_downgrade(canonical_name, sys_mm, sys_cuda):
    """Show Option 1: Install PyTorch matching nvcc."""
    print("\n" + "=" * 60)
    print(f"📦  OPTION 1: Install PyTorch matching your nvcc ({sys_mm})")
    print("=" * 60)
    print("\nTrade-offs:")
    print("  ✅ No system changes needed")
    print("  ✅ Faster to implement")
    print("  ❌ Older PyTorch version (may lack new features)")

    # Get PyTorch install command for this CUDA version
    torch_command = get_install_command("torch", sys_mm)

    print("\nCommands:")
    print("  # Uninstall current PyTorch")
    print("  pip uninstall torch torchvision torchaudio -y")
    print(f"\n  # Install PyTorch for CUDA {sys_mm}")
    if torch_command and torch_command != "Could not determine safe version.":
        print(f"  {torch_command}")
    else:
        print(f"  pip install torch --index-url https://download.pytorch.org/whl/cu{sys_mm.replace('.', '')}")
    print(f"\n  # Install {canonical_name}")
    print(f"  pip install {canonical_name} --no-build-isolation")


def _show_option2_nvcc_upgrade(canonical_name, torch_mm, torch_cuda, cuda_result):
    """Show Option 2: Upgrade nvcc to match PyTorch."""
    print("\n" + "=" * 60)
    print(f"⚙️   OPTION 2: Upgrade nvcc to match PyTorch ({torch_mm})")
    print("=" * 60)
    print("\nTrade-offs:")
    print("  ✅ Keep latest PyTorch")
    print("  ✅ Better long-term solution")
    print("  ❌ Requires system-level changes")
    print(f"  ❌ Verify driver supports CUDA {torch_mm}")

    # Check driver compatibility
    driver_detector = DetectorRegistry.get("nvidia_driver")
    driver_result = driver_detector.detect()

    if driver_result.detected:
        max_cuda = driver_result.metadata.get("max_cuda_version", "Unknown")
        if max_cuda != "Unknown":
            try:
                torch_float = float(torch_mm)
                max_float = float(max_cuda)
                if torch_float > max_float:
                    print(f"\n⚠️   WARNING: Your driver ({driver_result.version}) only supports CUDA {max_cuda}")
                    print(f"    PyTorch requires CUDA {torch_mm}")
                    print("    You may need to update your NVIDIA driver first!")
            except ValueError:
                pass

    # Check for WSL2
    wsl2_detector = DetectorRegistry.get("wsl2")
    if wsl2_detector and wsl2_detector.can_run():
        wsl2_result = wsl2_detector.detect()
        if wsl2_result.detected:
            print("\n⚠️   WSL2 DETECTED:")
            print("    Install CUDA inside WSL2, not Windows")
            print("    Guide: https://docs.nvidia.com/cuda/wsl-user-guide/")

    print("\nSteps:")
    print("  1. Check driver compatibility:")
    print("     env-doctor check")
    print(f"\n  2. Download CUDA Toolkit {torch_mm}:")
    print(f"     https://developer.nvidia.com/cuda-{torch_mm.replace('.', '-')}-0-download-archive")
    print("\n  3. Install CUDA Toolkit (follow NVIDIA's platform-specific guide)")
    print("\n  4. Verify installation:")
    print("     nvcc --version")
    print(f"\n  5. Install {canonical_name}:")
    print(f"     pip install {canonical_name} --no-build-isolation")
    print("\n" + "=" * 60)


def scan_command(output_json: bool = False):
    """
    Scan local directory for AI library imports.

    Args:
        output_json: Output as JSON (machine-readable)

    Note: This still uses legacy function as it's not environment detection.
    """
    libs = scan_imports_in_folder()

    if output_json:
        # JSON output
        output = {
            "status": "pass" if len(libs) > 0 else "fail",
            "timestamp": datetime.now().isoformat(),
            "dependencies": libs,
            "issues": [] if len(libs) > 0 else ["No common AI imports found"],
            "recommendations": [
                f"env-doctor install {lib}" for lib in libs if lib in ["torch", "tensorflow", "jax"]
            ]
        }
        print(json.dumps(output, indent=2))
        sys.exit(0)
    else:
        # Human output
        print("\n🔍  SCANNING CURRENT DIRECTORY...")
        if libs:
            print(f"Found imports for: {', '.join(libs)}")
            print("\nTo get safe install commands for these, run:")
            for lib in libs:
                if lib in ["torch", "tensorflow", "jax"]:
                    print(f"  env-doctor install {lib}")
        else:
            print("No common AI imports found.")


def debug_command():
    """
    NEW COMMAND: Debug mode that shows all detector results in detail.
    
    This is useful for troubleshooting and seeing raw detector output.
    """
    print("\n🔍  DEBUG MODE - Detailed Detector Information")
    print("=" * 60)
    
    # Get all registered detectors
    detector_names = DetectorRegistry.get_names()
    print(f"Registered Detectors: {', '.join(detector_names)}\n")
    
    # Run each detector and show results
    for name in detector_names:
        if name == "python_library":
            # Special case: python_library needs a library name
            continue
        
        print(f"\n--- {name.upper().replace('_', ' ')} ---")
        try:
            detector = DetectorRegistry.get(name)
            # CHECK if detector can run on this platform
            if not detector.can_run():
                print(f"Status: skipped (not applicable on {platform.system()})")
                continue

            result = detector.detect()
            
            print(f"Status: {result.status.value}")
            print(f"Component: {result.component}")
            if result.version:
                print(f"Version: {result.version}")
            if result.path:
                print(f"Path: {result.path}")
            if result.metadata:
                print(f"Metadata: {result.metadata}")
            if result.issues:
                print(f"Issues: {result.issues}")
            if result.recommendations:
                print(f"Recommendations: {result.recommendations}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Test python libraries separately
    print(f"\n--- PYTHON LIBRARIES ---")
    from .detectors.python_libraries import PythonLibraryDetector
    for lib in ["torch", "tensorflow", "jax"]:
        print(f"\n{lib}:")
        detector = PythonLibraryDetector(lib)
        result = detector.detect()
        print(f"  Status: {result.status.value}")
        if result.version:
            print(f"  Version: {result.version}")
        if result.metadata:
            print(f"  Metadata: {result.metadata}")




def print_cuda_detailed_info(cuda_result):
    """
    Print detailed CUDA toolkit information from comprehensive detection.
    
    Args:
        cuda_result: DetectionResult from CudaToolkitDetector
    """
    print("\n" + "="*60)
    print("🔧  DETAILED CUDA TOOLKIT ANALYSIS")
    print("="*60)
    
    if not cuda_result.detected:
        print("❌  No CUDA Toolkit detected")
        for rec in cuda_result.recommendations:
            print(f"    → {rec}")
        return
    
    # 1. Main version info
    print(f"\n📌  Primary CUDA Version: {cuda_result.version}")
    if cuda_result.path:
        print(f"    nvcc location: {cuda_result.path}")
    
    # 2. Installation count
    install_count = cuda_result.metadata.get("installation_count", 0)
    if install_count > 1:
        print(f"\n⚠️   Multiple Installations Detected: {install_count}")
        for i, inst in enumerate(cuda_result.metadata.get("installations", []), 1):
            print(f"    {i}. Version {inst['version']}: {inst['path']}")
    
    # 3. Environment Variables
    print("\n🔐  Environment Variables:")
    
    cuda_home = cuda_result.metadata.get("cuda_home", {})
    if cuda_home.get("status") == "set":
        print(f"    ✅ CUDA_HOME: {cuda_home['value']}")
    elif cuda_home.get("status") == "missing":
        print(f"    ❌ CUDA_HOME: Not set")
    elif cuda_home.get("status") == "invalid":
        print(f"    ❌ CUDA_HOME: {cuda_home['value']} (path doesn't exist)")
    
    # 4. PATH Configuration
    path_config = cuda_result.metadata.get("path_config", {})
    if path_config.get("correct"):
        print(f"    ✅ PATH: CUDA bin directory found")
    else:
        print(f"    ❌ PATH: CUDA bin directory missing")
        print(f"       {path_config.get('reason', 'Unknown issue')}")
    
    # 5. LD_LIBRARY_PATH (Linux only)
    if "ld_library_path" in cuda_result.metadata:
        ld_info = cuda_result.metadata["ld_library_path"]
        if ld_info.get("correct"):
            print(f"    ✅ LD_LIBRARY_PATH: CUDA lib directory found")
        else:
            print(f"    ❌ LD_LIBRARY_PATH: {ld_info.get('reason', 'Not configured')}")
    
    # 6. Runtime Library
    print("\n📚  Runtime Library:")
    libcudart = cuda_result.metadata.get("libcudart", {})
    if libcudart.get("found"):
        version = libcudart.get("version", "Unknown")
        print(f"    ✅ libcudart: Found (v{version})")
        print(f"       Location: {libcudart.get('path', 'Unknown')}")
    else:
        print(f"    ❌ libcudart: Not found")
    
    # 7. Driver Compatibility
    print("\n🖥️   Driver Compatibility:")
    driver_compat = cuda_result.metadata.get("driver_compatibility", {})
    if driver_compat.get("compatible"):
        print(f"    ✅ {driver_compat.get('message', 'Compatible')}")
        if "driver_version" in driver_compat:
            print(f"       Driver: {driver_compat['driver_version']}")
            print(f"       Max CUDA: {driver_compat['max_cuda']}")
    else:
        print(f"    ❌ {driver_compat.get('message', 'Incompatible')}")
    
    # 8. Issues & Recommendations
    if cuda_result.issues:
        print("\n⚠️   Issues Detected:")
        for issue in cuda_result.issues:
            print(f"    • {issue}")
    
    if cuda_result.recommendations:
        print("\n💡  Recommendations:")
        for rec in cuda_result.recommendations:
            print(f"    → {rec}")
    
    print("\n" + "="*60)



# New command: cuda-info
def cuda_info_command(output_json: bool = False):
    """
    Display comprehensive CUDA toolkit information.

    Args:
        output_json: Output as JSON (machine-readable)
    """
    cuda_detector = DetectorRegistry.get("cuda_toolkit")
    cuda_result = cuda_detector.detect()

    if output_json:
        print(json.dumps(cuda_result.to_dict(), indent=2))
        sys.exit(0 if cuda_result.status in [Status.SUCCESS, Status.WARNING] else 1)
    else:
        print_cuda_detailed_info(cuda_result)


def cuda_install_command(cuda_version: str = None):
    """
    Show step-by-step CUDA Toolkit installation instructions.

    If no version is specified, recommends one based on the GPU driver.

    Args:
        cuda_version: Optional specific CUDA version to install
    """
    from .utilities.platform_detect import detect_platform

    print("\n" + "=" * 60)
    print("CUDA TOOLKIT INSTALLATION GUIDE")
    print("=" * 60)

    # 1. Detect platform
    plat = detect_platform()
    print(f"\nDetected Platform:")
    if plat["is_wsl2"]:
        print(f"    WSL2 ({plat['distro']} {plat['distro_version']})")
    elif plat["os"] == "linux":
        print(f"    Linux ({plat['distro']} {plat['distro_version']}, {plat['arch']})")
    elif plat["os"] == "windows":
        print(f"    Windows ({plat['arch']})")
    else:
        print(f"    {plat['os']} ({plat['arch']})")

    # 2. Determine which CUDA version to recommend
    if cuda_version:
        recommended = cuda_version
        print(f"\nRequested CUDA Version: {recommended}")
    else:
        # Auto-detect from driver
        driver_detector = DetectorRegistry.get("nvidia_driver")
        driver_result = driver_detector.detect()

        if not driver_result.detected:
            print("\nNo NVIDIA driver detected.")
            print("    Install the NVIDIA driver first:")
            print("    https://www.nvidia.com/Download/index.aspx")
            return

        max_cuda = driver_result.metadata.get("max_cuda_version", "Unknown")
        print(f"\nDriver: {driver_result.version} (supports up to CUDA {max_cuda})")

        recommended = get_recommended_cuda_toolkit(max_cuda)
        if not recommended:
            print(f"\nNo installation instructions available for CUDA {max_cuda}")
            print(f"    Download manually: https://developer.nvidia.com/cuda-downloads")
            return

        print(f"Recommended CUDA Toolkit: {recommended}")

    # 3. Get installation steps
    install_info = get_cuda_install_steps(recommended, plat["platform_keys"])

    if not install_info:
        print(f"\nNo specific instructions for your platform.")
        print(f"    Download manually: https://developer.nvidia.com/cuda-downloads")

        # Show available platforms for this version
        versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
        version_data = versions.get(recommended, {})
        available = version_data.get("platforms", {})
        if available:
            print(f"\n    Available platforms for CUDA {recommended}:")
            for key, info in available.items():
                print(f"      - {info.get('label', key)}")
        return

    # 4. Print installation steps
    print(f"\n{'=' * 60}")
    print(f"{install_info.get('label', f'CUDA {recommended}')}")
    print(f"{'=' * 60}")

    # Prerequisites (if any)
    prereqs = install_info.get("prerequisites", [])
    if prereqs:
        print(f"\nPrerequisites:")
        for prereq in prereqs:
            print(f"    - {prereq}")

    # Installation steps
    print(f"\nInstallation Steps:")
    print(f"{'-' * 60}")
    for i, step in enumerate(install_info["steps"], 1):
        print(f"    {i}. {step}")

    # Post-install environment
    post_install = install_info.get("post_install", [])
    if post_install:
        print(f"\nPost-Installation Setup:")
        print(f"{'-' * 60}")
        for step in post_install:
            print(f"    {step}")

        persist = install_info.get("persist_env")
        if persist:
            print(f"\n    TIP: {persist}")

    # Verification
    verify = install_info.get("verify")
    if verify:
        print(f"\nVerify Installation:")
        print(f"{'-' * 60}")
        print(f"    {verify}")

    # Notes
    notes = install_info.get("notes")
    if notes:
        print(f"\nNotes:")
        print(f"    {notes}")

    # Download page link
    download = install_info.get("download_page")
    if download:
        print(f"\nOfficial Download Page:")
        print(f"    {download}")

    print(f"\n{'=' * 60}")
    print("After installation, run 'env-doctor check' to verify.")
    print(f"{'=' * 60}\n")


def print_cudnn_detailed_info(cudnn_result):
    """
    Print detailed cuDNN information from detection.

    Args:
        cudnn_result: DetectionResult from CudnnDetector
    """
    print("\n" + "="*60)
    print("🧠  DETAILED CUDNN ANALYSIS")
    print("="*60)

    if not cudnn_result.detected:
        print("❌  cuDNN library not found")
        for rec in cudnn_result.recommendations:
            print(f"    → {rec}")
        return

    # 1. Main version info
    print(f"\n📌  cuDNN Version: {cudnn_result.version}")
    if cudnn_result.path:
        print(f"    Primary Library: {cudnn_result.path}")

    # 2. Library count
    lib_count = cudnn_result.metadata.get("library_count", 1)
    if lib_count > 1:
        print(f"\n📚  Multiple cuDNN Libraries Found: {lib_count}")
        libraries = cudnn_result.metadata.get("libraries", [])
        for lib in libraries:
            print(f"    • {lib['path']}")

    # 3. Platform info
    platform_info = cudnn_result.metadata.get("platform", "Unknown")
    print(f"\n🔧  Platform: {platform_info}")

    # 4. Symlink/PATH status
    if platform_info == "Linux":
        symlink_status = cudnn_result.metadata.get("symlink_status", {})
        if symlink_status:
            print("\n🔗  Symlink Status:")
            if symlink_status.get("valid"):
                for symlink in symlink_status["valid"]:
                    print(f"    ✅ {symlink}")
            if symlink_status.get("missing"):
                for symlink in symlink_status["missing"]:
                    print(f"    ❌ Missing: {symlink}")
            if symlink_status.get("broken"):
                for symlink in symlink_status["broken"]:
                    print(f"    ⚠️  Broken: {symlink}")
    else:
        path_status = cudnn_result.metadata.get("path_status", {})
        if path_status:
            print("\n🔗  PATH Configuration:")
            if path_status.get("in_path"):
                print(f"    ✅ cuDNN DLL in PATH: {path_status.get('directory')}")
            else:
                print(f"    ❌ cuDNN DLL not in PATH")
                if path_status.get("suggested_path"):
                    print(f"       Suggested: {path_status.get('suggested_path')}")

    # 5. Multiple versions check
    multiple_versions = cudnn_result.metadata.get("multiple_versions")
    if multiple_versions and len(multiple_versions) > 1:
        print(f"\n⚠️   Multiple Versions Detected: {', '.join(multiple_versions)}")
        print("    Consider removing old versions to avoid conflicts")

    # 6. CUDA compatibility
    cuda_compat = cudnn_result.metadata.get("cuda_compatibility", {})
    if cuda_compat:
        print("\n🔗  CUDA Compatibility:")
        if cuda_compat.get("compatible"):
            print(f"    ✅ {cuda_compat.get('message', 'Compatible')}")
        else:
            print(f"    ❌ {cuda_compat.get('message', 'Incompatibility detected')}")

    # 7. Issues & Recommendations
    if cudnn_result.issues:
        print("\n⚠️   Issues Detected:")
        for issue in cudnn_result.issues:
            print(f"    • {issue}")

    if cudnn_result.recommendations:
        print("\n💡  Recommendations:")
        for rec in cudnn_result.recommendations:
            print(f"    → {rec}")

    print("\n" + "="*60)


def cudnn_info_command(output_json: bool = False):
    """
    Display comprehensive cuDNN library information.

    Args:
        output_json: Output as JSON (machine-readable)
    """
    cudnn_detector = DetectorRegistry.get("cudnn")
    if not cudnn_detector.can_run():
        if output_json:
            print(json.dumps({"error": "cuDNN detector not supported on this platform"}))
        else:
            print("❌  cuDNN detector not supported on this platform")
        sys.exit(1)
        return

    cudnn_result = cudnn_detector.detect()

    if output_json:
        print(json.dumps(cudnn_result.to_dict(), indent=2))
        sys.exit(0 if cudnn_result.status in [Status.SUCCESS, Status.WARNING] else 1)
    else:
        print_cudnn_detailed_info(cudnn_result)


def dockerfile_command(dockerfile_path: str = "Dockerfile"):
    """
    Validate a Dockerfile for GPU/CUDA configuration issues.

    Args:
        dockerfile_path: Path to Dockerfile (default: ./Dockerfile)
    """
    from .validators.dockerfile_validator import DockerfileValidator

    print(f"\n🐳  DOCKERFILE VALIDATION: {dockerfile_path}")
    print("="*60)

    validator = DockerfileValidator(dockerfile_path)
    result = validator.validate()

    _print_validation_result(result)

    # Exit with error code if errors found
    sys.exit(1 if result.error_count > 0 else 0)


def docker_compose_command(compose_path: str = "docker-compose.yml"):
    """
    Validate a docker-compose.yml for GPU configuration issues.

    Args:
        compose_path: Path to docker-compose.yml (default: ./docker-compose.yml)
    """
    from .validators.compose_validator import ComposeValidator

    print(f"\n🐳  DOCKER COMPOSE VALIDATION: {compose_path}")
    print("="*60)

    validator = ComposeValidator(compose_path)
    result = validator.validate()

    _print_validation_result(result)

    # Exit with error code if errors found
    sys.exit(1 if result.error_count > 0 else 0)


def _print_validation_result(result):
    """
    Print a validation result with colorized, grouped output.

    Args:
        result: ValidationResult object
    """
    from .validators.models import Severity

    if not result.issues:
        print("\n✅  No issues found! Configuration looks good.")
        return

    # Print issues grouped by severity
    for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO]:
        issues = result.get_issues_by_severity(severity)
        if not issues:
            continue

        # Severity header with emoji
        if severity == Severity.ERROR:
            print(f"\n❌  ERRORS ({len(issues)}):")
        elif severity == Severity.WARNING:
            print(f"\n⚠️   WARNINGS ({len(issues)}):")
        else:
            print(f"\nℹ️   INFO ({len(issues)}):")

        print("-" * 60)

        # Print each issue
        for issue in issues:
            if issue.line_number > 0:
                print(f"\nLine {issue.line_number}:")
            else:
                print()

            print(f"  Issue: {issue.issue}")
            print(f"  Fix:   {issue.recommendation}")

            if issue.corrected_command:
                print(f"\n  Suggested fix:")
                for line in issue.corrected_command.split('\n'):
                    print(f"    {line}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  ❌ Errors:   {result.error_count}")
    print(f"  ⚠️  Warnings: {result.warning_count}")
    print(f"  ℹ️  Info:     {result.info_count}")

    if result.error_count > 0:
        print("\n❌  Validation FAILED. Fix errors before deploying.")
    elif result.warning_count > 0:
        print("\n⚠️   Validation passed with warnings. Review before deploying.")
    else:
        print("\n✅  All checks passed!")
def model_command(model_name: str, precision: str = None):
    """
    Check if a model can run on available hardware.

    Args:
        model_name: Name of the model to check
        precision: Optional specific precision to check
    """
    checker = ModelChecker()
    result = checker.check_compatibility(model_name, precision)

    if not result["success"]:
        print(f"\n❌  {result['error']}")
        if result.get("suggestions"):
            print(f"\n💡  Did you mean:")
            for sugg in result["suggestions"]:
                print(f"    • {sugg}")
            print(f"\n    Run 'env-doctor model --list' to see all available models")
        print()
        return
    
    print_model_compatibility(result)


def print_model_compatibility(result: dict):
    """
    Pretty-print model compatibility analysis.

    Args:
        result: Compatibility check result from ModelChecker
    """
    model_info = result["model_info"]
    gpu_info = result["gpu_info"]
    vram_reqs = result["vram_requirements"]
    compat = result["compatibility"]
    recs = result["recommendations"]

    # Header
    print(f"\n🤖  Checking: {result['model_name'].upper()}")
    if result.get("fetched_from_hf"):
        print(f"    (Fetched from HuggingFace API - cached for future use)")
    print(f"    Parameters: {model_info['params_b']}B")
    if model_info.get("hf_id"):
        print(f"    HuggingFace: {model_info['hf_id']}")

    # GPU Info
    print(f"\n🖥️   Your Hardware:")
    if gpu_info["available"]:
        if gpu_info["gpu_count"] == 1:
            print(
                f"    {gpu_info['primary_gpu_name']} "
                f"({gpu_info['primary_gpu_vram_mb'] // 1024}GB VRAM)"
            )
        else:
            print(f"    {gpu_info['gpu_count']}x {gpu_info['primary_gpu_name']}")
            print(f"    Total VRAM: {gpu_info['total_vram_mb'] // 1024}GB")
    else:
        print(f"    ❌ No NVIDIA GPU detected")

    # Compatibility Table
    print(f"\n" + "=" * 60)
    print(f"💾  VRAM Requirements & Compatibility")
    print("=" * 60)

    if compat.get("no_gpu_available"):
        print("\n❌  No GPU available - cannot run this model locally\n")
    else:
        fits_any = False

        for precision in ["fp32", "fp16", "bf16", "int8", "int4", "fp8"]:
            if precision not in vram_reqs:
                continue

            req_info = vram_reqs[precision]
            fit_info = compat["fits_on_single_gpu"][precision]

            required_mb = req_info["vram_mb"]
            required_gb = required_mb / 1024
            source = req_info["source"]
            source_indicator = "" if source == "measured" else "~"

            if fit_info["fits"]:
                free_gb = fit_info["free_vram_mb"] / 1024
                print(
                    f"  ✅  {precision.upper():5s}: "
                    f"{source_indicator}{required_gb:6.1f}GB ({source:9s}) - "
                    f"{free_gb:5.1f}GB free"
                )
                fits_any = True
            else:
                shortage_gb = fit_info["shortage_mb"] / 1024
                print(
                    f"  ❌  {precision.upper():5s}: "
                    f"{source_indicator}{required_gb:6.1f}GB ({source:9s}) - "
                    f"Need {shortage_gb:5.1f}GB more"
                )

        # Overall status
        print("\n" + "=" * 60)
        if fits_any:
            print("✅  This model WILL FIT on your GPU!\n")
        else:
            print("❌  This model WON'T FIT on your GPU\n")

    # Recommendations
    if recs:
        print("💡  Recommendations:")
        print("=" * 60)
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec}")
        print()

    # Reference
    if model_info.get("hf_id"):
        print("=" * 60)
        print("📚  Reference:")
        print(f"    https://huggingface.co/{model_info['hf_id']}")
        print("=" * 60)


def list_models_command():
    """
    List all available models in database.
    """
    from .utilities import VRAMCalculator

    calc = VRAMCalculator()
    models_by_category = calc.list_all_models()

    print("\n📋  Available Models in Database")
    print("=" * 60)

    category_names = {
        "llm": "🧠  Large Language Models (LLMs)",
        "diffusion": "🎨  Diffusion Models (Image Generation)",
        "audio": "🔊  Audio Models (Speech Recognition)",
        "language": "📝  Language Models (Text Processing)",
    }

    for category in ["llm", "diffusion", "audio", "language"]:
        if category not in models_by_category:
            continue

        print(f"\n{category_names.get(category, category.upper())}")
        print("-" * 60)

        for model in sorted(
            models_by_category[category], key=lambda x: x["params_b"]
        ):
            name = model["name"]
            params = model["params_b"]
            print(f"  • {name:30s} ({params:6.1f}B params)")

    print("\n" + "=" * 60)
    print("💡  Usage:")
    print("    env-doctor model <model-name>")
    print("    env-doctor model <model-name> --precision int4")
    print("\n📖  To add models:")
    print("    See docs/ADDING_MODELS.md")
    print("=" * 60 + "\n")


# Update main() to add new command
def main():
    """Main entry point with argument parsing."""
    # Enable UTF-8 output on Windows
    import sys
    if sys.platform == "win32":
        # Configure stdout to use UTF-8
        import io
        if sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="env-doctor: The AI Environment Fixer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  env-doctor check              # Diagnose your environment
  env-doctor cuda-info          # Detailed CUDA toolkit analysis
  env-doctor cuda-install       # Step-by-step CUDA installation guide
  env-doctor cuda-install 12.4  # Install specific CUDA version
  env-doctor cudnn-info         # Detailed cuDNN library analysis
  env-doctor dockerfile         # Validate Dockerfile for GPU issues
  env-doctor docker-compose     # Validate docker-compose.yml for GPU issues
  env-doctor model llama-3-8b   # Check if model fits on your GPU
  env-doctor model --list       # List all available models
  env-doctor python-compat      # Check Python version compatibility
  env-doctor install torch      # Get safe install command for PyTorch
  env-doctor scan               # Scan project for AI library imports
  env-doctor debug              # Show detailed detector information
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Diagnose environment compatibility"
    )
    check_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (machine-readable)'
    )
    check_parser.add_argument(
        '--ci',
        action='store_true',
        help='CI-friendly mode (implies --json with proper exit codes)'
    )

    # CUDA Info command (NEW)
    cuda_info_parser = subparsers.add_parser(
        "cuda-info",
        help="Detailed CUDA toolkit analysis"
    )
    cuda_info_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (machine-readable)'
    )

    # CUDA Install command (NEW)
    cuda_install_parser = subparsers.add_parser(
        "cuda-install",
        help="Step-by-step CUDA Toolkit installation guide"
    )
    cuda_install_parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help="Specific CUDA version to install (default: auto-detect from driver)"
    )

    # cuDNN Info command (NEW)
    cudnn_info_parser = subparsers.add_parser(
        "cudnn-info",
        help="Detailed cuDNN library analysis"
    )
    cudnn_info_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (machine-readable)'
    )

    # Dockerfile validation command (NEW)
    dockerfile_p = subparsers.add_parser(
        "dockerfile",
        help="Validate Dockerfile for GPU/CUDA configuration issues"
    )
    dockerfile_p.add_argument(
        "path",
        nargs="?",
        default="Dockerfile",
        help="Path to Dockerfile (default: ./Dockerfile)"
    )

    # Docker Compose validation command (NEW)
    compose_p = subparsers.add_parser(
        "docker-compose",
        help="Validate docker-compose.yml for GPU configuration issues"
    )
    compose_p.add_argument(
        "path",
        nargs="?",
        default="docker-compose.yml",
        help="Path to docker-compose.yml (default: ./docker-compose.yml)"
    )

    # Python Compatibility command
    python_compat_parser = subparsers.add_parser(
        "python-compat",
        help="Check Python version compatibility with AI libraries"
    )
    python_compat_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (machine-readable)'
    )

    # Install command
    install_p = subparsers.add_parser(
        "install",
        help="Get safe installation command for a library"
    )
    install_p.add_argument(
        "library",
        help="Library name (e.g., torch, tensorflow, jax)"
    )

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan local files for AI library imports"
    )
    scan_parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (machine-readable)'
    )
    
    # Debug command
    subparsers.add_parser(
        "debug",
        help="Show detailed detector information (for troubleshooting)"
    )

    # Model command
    model_p = subparsers.add_parser(
        "model",
        help="Check if AI model fits on your GPU"
    )
    model_p.add_argument(
        "model_name",
        nargs="?",
        help="Model name (e.g., llama-3-8b, stable-diffusion-xl)"
    )
    model_p.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "int8", "int4", "fp8"],
        help="Check specific precision (default: show all)"
    )
    model_p.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.command == "check":
        check_command(
            output_json=getattr(args, 'json', False),
            ci=getattr(args, 'ci', False)
        )
    elif args.command == "cuda-info":
        cuda_info_command(
            output_json=getattr(args, 'json', False)
        )
    elif args.command == "cuda-install":
        cuda_install_command(getattr(args, 'version', None))
    elif args.command == "cudnn-info":
        cudnn_info_command(
            output_json=getattr(args, 'json', False)
        )
    elif args.command == "dockerfile":
        dockerfile_command(args.path)
    elif args.command == "docker-compose":
        docker_compose_command(args.path)
    elif args.command == "python-compat":
        python_compat_command(
            output_json=getattr(args, 'json', False)
        )
    elif args.command == "model":
        if args.list:
            list_models_command()
        elif args.model_name:
            model_command(args.model_name, args.precision)
        else:
            model_p.print_help()
    elif args.command == "install":
        install_command(args.library)
    elif args.command == "scan":
        scan_command(
            output_json=getattr(args, 'json', False)
        )
    elif args.command == "debug":
        debug_command()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()