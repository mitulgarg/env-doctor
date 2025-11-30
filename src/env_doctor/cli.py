import sys
import os
import argparse
from .checks import (
    get_nvidia_driver_version, 
    get_system_cuda_version, 
    get_installed_library_version,
    scan_imports_in_folder,
    check_broken_imports
)
from .db import get_max_cuda_for_driver, get_install_command, DB_DATA

def check_compilation_health(sys_cuda, torch_cuda):
    print("\n🏭  COMPILATION HEALTH (For Flash-Attention/AutoGPTQ)")
    if not sys_cuda:
        print("⚠️   System CUDA (nvcc) NOT found.")
        print("    -> You cannot install 'flash-attention' or 'auto-gptq' from source.")
        return
    if torch_cuda == "Unknown":
        print("❓  Torch CUDA version unknown. Skipping check.")
        return

    sys_mm = ".".join(sys_cuda.split(".")[:2])
    torch_mm = ".".join(torch_cuda.split(".")[:2])

    if sys_mm == torch_mm:
        print(f"✅  PERFECT SYMMETRY: System ({sys_cuda}) == Torch ({torch_cuda})")
    else:
        print(f"❌  ASYMMETRY DETECTED: System ({sys_cuda}) != Torch ({torch_cuda})")
        print("    -> pip install flash-attention will likely FAIL.")

def check_system_path():
    print("\n🔗  SYSTEM LINKING (For TensorFlow/JAX)")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        print("⚠️   LD_LIBRARY_PATH is unset.")
        return
    print(f"    LD_LIBRARY_PATH: {ld_path}")
    if "cuda" not in ld_path.lower():
        print("⚠️   Warning: LD_LIBRARY_PATH is set but does not seem to point to any CUDA folders.")

def check_command():
    print("\n🩺  ENV-DOCTOR DIAGNOSIS  🩺")
    print("==============================")

    # --- Show DB Status ---
    meta = DB_DATA.get("_metadata", {})
    if meta:
        print(f"🛡️  DB Verified: {meta.get('last_verified', 'Unknown')}")
        print(f"    Method: {meta.get('method', 'Unknown')}")
        print("------------------------------")
    
    # 1. Hardware
    driver = get_nvidia_driver_version()
    if driver:
        max_cuda = get_max_cuda_for_driver(driver)
        print(f"✅  GPU Driver Found: {driver}")
        print(f"    -> Max Supported CUDA: {max_cuda}")
    else:
        print("⚠️   No NVIDIA Driver detected via NVML.")
        max_cuda = None

    # 2. System
    sys_cuda = get_system_cuda_version()
    if sys_cuda:
        print(f"✅  System CUDA (nvcc): {sys_cuda}")
    else:
        print("ℹ️   System CUDA (nvcc) not found.")

    print("------------------------------")

    # 3. Software
    libs = ["torch", "tensorflow", "jax"]
    torch_cuda_version = None

    for lib in libs:
        info = get_installed_library_version(lib)
        if info:
            print(f"📦  Found {lib}: v{info['version']}")
            if info['cuda'] != "Unknown":
                print(f"    -> Bundled CUDA: {info['cuda']}")
                if lib == "torch": torch_cuda_version = info['cuda']
                
                if max_cuda:
                    try:
                        cuda_num = info['cuda'].split(" ")[0].replace("x", "0")
                        if float(cuda_num) > float(max_cuda):
                            print(f"    ❌ CRITICAL CONFLICT: Lib uses {info['cuda']}, Driver supports {max_cuda}!")
                            print(f"    Run 'doctor install {lib}' to fix.")
                        else:
                            print(f"    ✅ Compatible with Driver.")
                    except ValueError: pass
            else:
                print(f"    -> Bundled CUDA: Not Detected")
        else:
            print(f"❌  {lib} is NOT installed.")

    if torch_cuda_version:
        check_compilation_health(sys_cuda, torch_cuda_version)
    
    check_system_path()
    check_broken_imports()

def install_command(package_name):
    print(f"\n🩺  PRESCRIPTION FOR: {package_name}")
    driver = get_nvidia_driver_version()
    if not driver:
        print("⚠️  No NVIDIA Driver found. Assuming CPU-only.")
        print(f"   pip install {package_name}")
        return

    max_cuda = get_max_cuda_for_driver(driver)
    print(f"Detected Driver: {driver} (Supports up to CUDA {max_cuda})")
    command = get_install_command(package_name, max_cuda)
    print("\n⬇️   Run this command:")
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
            if lib in ["torch", "tensorflow", "jax"]:
                print(f"  env-doctor install {lib}")
    else:
        print("No common AI imports found.")

def main():
    parser = argparse.ArgumentParser(description="env-doctor: The AI Environment Fixer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("check", help="Diagnose environment.")
    
    install_p = subparsers.add_parser("install", help="Get safe install command.")
    install_p.add_argument("library", help="Library name (e.g., torch)")

    subparsers.add_parser("scan", help="Scan local files.")

    args = parser.parse_args()

    if args.command == "check": check_command()
    elif args.command == "install": install_command(args.library)
    elif args.command == "scan": scan_command()
    else: parser.print_help()

if __name__ == "__main__":
    main()