import sys
import os
import argparse
from .checks import (
    get_nvidia_driver_version, 
    get_system_cuda_version, 
    get_installed_library_version,
    scan_imports_in_folder
)
from .db import get_max_cuda_for_driver, get_install_command

#For add on stuff like Flash-Attention  because their compilation depends on nvcc, the pytorch included cuda version needs to match 
def check_compilation_health(sys_cuda, torch_cuda):
    """
    Verifies if the user can compile custom kernels (Flash-Attention, AutoGPTQ).
    Rule: System CUDA (nvcc) must match Torch Bundled CUDA.
    """
    print("\n🏭  COMPILATION HEALTH (For Flash-Attention/AutoGPTQ)")
    
    if not sys_cuda:
        print("⚠️   System CUDA (nvcc) NOT found.")
        print("    -> You cannot install 'flash-attention' or 'auto-gptq' from source.")
        return

    if torch_cuda == "Unknown":
        print("❓  Torch CUDA version unknown. Skipping check.")
        return

    # Simple major.minor comparison (e.g., 11.8 vs 11.8)
    # We strip patch versions just in case (11.8.0 vs 11.8)
    sys_major_minor = ".".join(sys_cuda.split(".")[:2])
    torch_major_minor = ".".join(torch_cuda.split(".")[:2])

    if sys_major_minor == torch_major_minor:
        print(f"✅  PERFECT SYMMETRY: System ({sys_cuda}) == Torch ({torch_cuda})")
        print("    -> You can compile custom extensions safely.")
    else:
        print(f"❌  ASYMMETRY DETECTED: System ({sys_cuda}) != Torch ({torch_cuda})")
        print("    -> pip install flash-attention will likely FAIL due to header mismatch.")
        print("    -> Fix: Install a PyTorch version that matches your system 'nvcc', or update 'nvcc'.")

def check_system_path():
    """
    Checks for the 'TensorFlow Nightmare' (LD_LIBRARY_PATH).
    """
    print("\n🔗  SYSTEM LINKING (For TensorFlow/JAX)")
    
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        print("⚠️   LD_LIBRARY_PATH is unset.")
        print("    -> If TensorFlow crashes with 'libcudart.so not found', you typically need to set this.")
        print("    -> Example: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        return

    print(f"    LD_LIBRARY_PATH: {ld_path}")
    if "cuda" not in ld_path.lower():
        print("⚠️   Warning: LD_LIBRARY_PATH is set but does not seem to point to any CUDA folders.")

def check_command():
    print("\n🩺  ENV-DOCTOR DIAGNOSIS  🩺")
    print("==============================")


    # --- Show DB Status ---
    '''
    meta = DB_DATA.get("_metadata", {})
    if meta:
        print(f"🛡️  DB Verified: {meta.get('last_verified', 'Unknown')}")
        print(f"    Method: {meta.get('method', 'Unknown')}")
        print("------------------------------")
    '''
    
    # 1. Hardware Check
    driver = get_nvidia_driver_version()
    if driver:
        max_cuda = get_max_cuda_for_driver(driver)
        print(f"✅  GPU Driver Found: {driver}")
        print(f"    -> Max Supported CUDA: {max_cuda}")
    else:
        print("⚠️   No NVIDIA Driver detected via NVML.")
        max_cuda = None

    # 2. System Check
    sys_cuda = get_system_cuda_version()
    if sys_cuda:
        print(f"✅  System CUDA (nvcc): {sys_cuda}")
    else:
        print("ℹ️   System CUDA (nvcc) not found.")

    print("------------------------------")

    # 3. Software Check
    libs_to_check = ["torch", "tensorflow", "jax"]
    torch_cuda_version = None

    for lib in libs_to_check:
        info = get_installed_library_version(lib)
        if info:
            print(f"📦  Found {lib}: v{info['version']}")
            
            # Check CUDA Bundle
            if info['cuda'] and info['cuda'] != "Unknown":
                print(f"    -> Bundled CUDA: {info['cuda']}")
                if lib == "torch": 
                    torch_cuda_version = info['cuda']

                # LOGIC: Check mismatch against Driver
                if max_cuda:
                    try:
                        # Simple float comparison
                        if float(info['cuda']) > float(max_cuda):
                            print(f"    ❌ CRITICAL CONFLICT: Lib requires CUDA {info['cuda']}, but Driver only supports {max_cuda}!")
                        else:
                            print(f"    ✅ Compatible with Driver.")
                    except ValueError:
                        pass 
            else:
                print(f"    -> Bundled CUDA: Not Detected (CPU version or System-Linked)")
        else:
            print(f"❌  {lib} is NOT installed.")

    # 4. Advanced Checks
    # Only run these if we actually found tools to check against
    if torch_cuda_version:
        check_compilation_health(sys_cuda, torch_cuda_version)
    
    check_system_path()

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
    print(f"Detected Driver: {driver} (Supports up to CUDA {max_cuda})")
    
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