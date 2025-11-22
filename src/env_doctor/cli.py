import sys
import argparse
from .checks import (
    get_nvidia_driver_version, 
    get_system_cuda_version, 
    get_installed_library_version,
    scan_imports_in_folder
)
from .db import get_max_cuda_for_driver, get_install_command, DRIVER_TO_CUDA

def check_command():
    print("\n🩺  ENV-DOCTOR DIAGNOSIS  🩺")
    print("==============================")
    
    # 1. Hardware Check
    driver = get_nvidia_driver_version()
    if driver:
        max_cuda = get_max_cuda_for_driver(driver)
        print(f"✅  GPU Driver Found: {driver}")
        print(f"    -> Max Supported CUDA: {max_cuda}")
    else:
        print("⚠️   No NVIDIA Driver detected via NVML.")
        print("    (Ensure you are running on a machine with an NVIDIA GPU)")
        max_cuda = None

    # 2. System Check
    sys_cuda = get_system_cuda_version()
    if sys_cuda:
        print(f"✅  System CUDA (nvcc): {sys_cuda}")
    else:
        print("ℹ️   System CUDA (nvcc) not found (Not critical for runtime, needed for compilation).")

    print("------------------------------")

    # 3. Software Check
    libs_to_check = ["torch", "tensorflow"]
    for lib in libs_to_check:
        info = get_installed_library_version(lib)
        if info:
            print(f"📦  Found {lib}: v{info['version']}")
            if info['cuda'] and info['cuda'] != "Unknown":
                print(f"    -> Bundled CUDA: {info['cuda']}")
                
                # LOGIC: Check mismatch
                if max_cuda:
                    # Simple float comparison
                    try:
                        if float(info['cuda']) > float(max_cuda):
                            print(f"    ❌ CRITICAL CONFLICT: Lib requires CUDA {info['cuda']}, but Driver only supports {max_cuda}!")
                        else:
                            print(f"    ✅ Compatible with Driver.")
                    except ValueError:
                        pass # Version string parsing failed
            else:
                print(f"    -> Bundled CUDA: Not Detected (CPU version?)")
        else:
            print(f"❌  {lib} is NOT installed.")

def install_command(package_name):
    print(f"\n🩺  PRESCRIPTION FOR: {package_name}")
    print("====================================")
    
    driver = get_nvidia_driver_version()
    if not driver:
        print("⚠️  No NVIDIA Driver found. Assuming CPU-only or Driverless setup.")
        print("Recommendation: Standard install")
        print(f"   pip install {package_name}")
        return

    max_cuda = get_max_cuda_for_driver(driver)
    print(f"detected Driver: {driver} (Supports up to CUDA {max_cuda})")
    
    command = get_install_command(package_name, max_cuda)
    
    print("\n⬇️   Run this command to install the SAFE version:")
    print("---------------------------------------------------")
    print(command)
    print("---------------------------------------------------")

def scan_command():
    print("\n🔍  SCANNING CURRENT DIRECTORY...")
    libs = scan_imports_in_folder()
    if libs:
        print(f"Found imports for: {', '.join(libs)}")
        print("\nTo get safe install commands for these, run:")
        for lib in libs:
            if lib in ["torch", "tensorflow"]:
                print(f"  env-doctor install {lib}")
    else:
        print("No common AI imports found in .py files.")

def main():
    parser = argparse.ArgumentParser(description="env-doctor: The AI Environment Fixer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: check
    subparsers.add_parser("check", help="Diagnose current environment hardware & software.")

    # Subcommand: install
    install_parser = subparsers.add_parser("install", help="Get safe install command for a library.")
    install_parser.add_argument("library", help="Name of library (e.g., torch, tensorflow)")

    # Subcommand: scan
    subparsers.add_parser("scan", help="Scan local files to detect needed libraries.")

    args = parser.parse_args()

    if args.command == "check":
        check_command()
    elif args.command == "install":
        install_command(args.library)
    elif args.command == "scan":
        scan_command()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()