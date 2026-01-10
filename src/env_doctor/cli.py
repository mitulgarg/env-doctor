"""
Modernized CLI using the detector architecture.

This version demonstrates how to migrate from procedural checks
to the new detector-based system.
"""
import sys
import os
import argparse
import platform 
from .core.registry import DetectorRegistry
from .core.detector import Status
from .db import get_max_cuda_for_driver, get_install_command, DB_DATA

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
        print("    -> pip install flash-attention will likely FAIL.")
        print(f"    → Consider installing CUDA Toolkit {torch_mm}")


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


def check_command():
    """
    Main diagnostic command using detector architecture.
    
    This is the MODERNIZED version that uses DetectorRegistry
    instead of direct function calls.
    """
    print("\n🩺  ENV-DOCTOR DIAGNOSIS  🩺")
    print("==============================")

    # --- Show DB Status ---
    meta = DB_DATA.get("_metadata", {})
    if meta:
        print(f"🛡️  DB Verified: {meta.get('last_verified', 'Unknown')}")
        print(f"    Method: {meta.get('method', 'Unknown')}")
        print("------------------------------")
    
    # === STEP 1: Environment Detection ===
    # Use the WSL2Detector to check environment first
    wsl2_detector = DetectorRegistry.get("wsl2")
    if wsl2_detector.can_run():
        wsl2_result = wsl2_detector.detect()
        print_detection_result(wsl2_result, "🐧")
        print("------------------------------")
    
    # === STEP 2: Hardware Detection ===
    # Use the new NvidiaDriverDetector
    driver_detector = DetectorRegistry.get("nvidia_driver")
    driver_result = driver_detector.detect()
    
    if driver_result.detected:
        max_cuda = driver_result.metadata.get("max_cuda_version", "Unknown")
        print(f"✅  GPU Driver Found: {driver_result.version}")
        print(f"    → Max Supported CUDA: {max_cuda}")
        print(f"    → Detection Method: {driver_result.metadata.get('detection_method', 'unknown')}")
    else:
        print("⚠️   No NVIDIA Driver detected.")
        for rec in driver_result.recommendations:
            print(f"    → {rec}")
        max_cuda = None

    # === STEP 3: System CUDA Detection (ENHANCED) ===
    cuda_detector = DetectorRegistry.get("cuda_toolkit")
    cuda_result = cuda_detector.detect()
    
    # Print basic summary
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

    #cuDNN Detection
    # In check_command(), after CUDA check
    cudnn_detector = DetectorRegistry.get("cudnn")
    if cudnn_detector.can_run():
        cudnn_result = cudnn_detector.detect()
        if cudnn_result.detected:
            print(f"✅  cuDNN: v{cudnn_result.version}")

    # === STEP 4: Python Libraries Detection ===
    # Use the new PythonLibraryDetector for each library
    libs = ["torch", "tensorflow", "jax"]
    torch_result = None
    
    for lib in libs:
        # Import here to avoid circular imports
        from .detectors.python_libraries import PythonLibraryDetector
        
        lib_detector = PythonLibraryDetector(lib)
        lib_result = lib_detector.detect()
        
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
            
            # Store torch result for compilation check
            if lib == "torch":
                torch_result = lib_result
        else:
            print(f"❌  {lib} is NOT installed.")

    # === STEP 5: Compilation Health Check ===
    if torch_result and torch_result.detected:
        check_compilation_health(cuda_result, torch_result)
    
    # === STEP 6: System Path Check ===
    check_system_path()
    
    # === STEP 7: Code Migration Check ===
    # (Not yet refactored - still using legacy function)
    check_broken_imports()

    # === STEP 8: Offer detailed analysis ===
    if cuda_result.detected and (cuda_result.issues or cuda_result.metadata.get("installation_count", 1) > 1):
        print("\n💡  TIP: Run 'env-doctor cuda-info' for detailed CUDA analysis")



def install_command(package_name):
    """
    Provide installation prescription for a package.
    
    Uses NvidiaDriverDetector to determine compatible CUDA version.
    """
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


def scan_command():
    """
    Scan local directory for AI library imports.
    
    Note: This still uses legacy function as it's not environment detection.
    """
    print("\n🔍  SCANNING CURRENT DIRECTORY...")
    libs = scan_imports_in_folder()
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
def cuda_info_command():
    """
    Display comprehensive CUDA toolkit information.
    """
    cuda_detector = DetectorRegistry.get("cuda_toolkit")
    cuda_result = cuda_detector.detect()

    print_cuda_detailed_info(cuda_result)


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


def cudnn_info_command():
    """
    Display comprehensive cuDNN library information.
    """
    cudnn_detector = DetectorRegistry.get("cudnn")
    if not cudnn_detector.can_run():
        print("❌  cuDNN detector not supported on this platform")
        return

    cudnn_result = cudnn_detector.detect()
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


# Update main() to add new command
def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="env-doctor: The AI Environment Fixer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  env-doctor check              # Diagnose your environment
  env-doctor cuda-info          # Detailed CUDA toolkit analysis
  env-doctor cudnn-info         # Detailed cuDNN library analysis
  env-doctor dockerfile         # Validate Dockerfile for GPU issues
  env-doctor docker-compose     # Validate docker-compose.yml for GPU issues
  env-doctor install torch      # Get safe install command for PyTorch
  env-doctor scan               # Scan project for AI library imports
  env-doctor debug              # Show detailed detector information
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    subparsers.add_parser(
        "check",
        help="Diagnose environment compatibility"
    )

    # CUDA Info command (NEW)
    subparsers.add_parser(
        "cuda-info",
        help="Detailed CUDA toolkit analysis"
    )

    # cuDNN Info command (NEW)
    subparsers.add_parser(
        "cudnn-info",
        help="Detailed cuDNN library analysis"
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
    subparsers.add_parser(
        "scan", 
        help="Scan local files for AI library imports"
    )
    
    # Debug command
    subparsers.add_parser(
        "debug",
        help="Show detailed detector information (for troubleshooting)"
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.command == "check":
        check_command()
    elif args.command == "cuda-info":
        cuda_info_command()
    elif args.command == "cudnn-info":
        cudnn_info_command()
    elif args.command == "dockerfile":
        dockerfile_command(args.path)
    elif args.command == "docker-compose":
        docker_compose_command(args.path)
    elif args.command == "install":
        install_command(args.library)
    elif args.command == "scan":
        scan_command()
    elif args.command == "debug":
        debug_command()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()