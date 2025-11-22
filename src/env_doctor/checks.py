import sys
import shutil
import subprocess
import importlib
import os
import glob
import re

# Try to import NVML, handle failure if not installed
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

def get_nvidia_driver_version():
    """
    Leg 1: Hardware Check.
    Returns the NVIDIA Driver Version as a string (e.g., '535.129') or None.
    """
    if not HAS_NVML:
        return None
    
    try:
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
        pynvml.nvmlShutdown()
        return driver
    except Exception as e:
        # Could happen if no GPU is present
        return None

def get_system_cuda_version():
    """
    Leg 2: System Compiler Check.
    Checks `nvcc --version`. Returns string (e.g., '11.8') or None.
    """
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        # Try common Linux path
        if os.path.exists("/usr/local/cuda/bin/nvcc"):
            nvcc_path = "/usr/local/cuda/bin/nvcc"
        else:
            return None
            
    try:
        result = subprocess.check_output([nvcc_path, "--version"], encoding="utf-8")
        # Output format usually: "... release 11.8, V11.8.89 ..."
        match = re.search(r"release (\d+\.\d+)", result)
        if match:
            return match.group(1)
    except Exception:
        return None
    return None

def get_installed_library_version(lib_name):
    """
    Leg 3: Python Library Check.
    Returns dictionary with 'version' and 'cuda_version' (if detected).
    """
    try:
        lib = importlib.import_module(lib_name)
        version = getattr(lib, "__version__", "Unknown")
        cuda_ver = "Unknown"
        
        if lib_name == "torch":
            # PyTorch stores CUDA version in torch.version.cuda
            try:
                cuda_ver = lib.version.cuda
            except AttributeError:
                pass
        
        elif lib_name == "tensorflow":
            # TF is harder, often needs a live GPU call or build_info
            try:
                # This works in newer TF versions
                sys_config = getattr(lib, "sysconfig", None)
                if sys_config and hasattr(sys_config, "get_build_info"):
                    cuda_ver = sys_config.get_build_info().get("cuda_version", "Unknown")
            except Exception:
                pass

        return {"version": version, "cuda": cuda_ver}
    except ImportError:
        return None

def scan_imports_in_folder(folder_path="."):
    """
    Scans all .py files in the directory to find 'import torch', 'import tensorflow', etc.
    """
    found_libs = set()
    # Simple regex to catch 'import x' or 'from x import y'
    import_regex = re.compile(r"^\s*(?:import|from)\s+(\w+)")
    
    # Walk through current directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            match = import_regex.match(line)
                            if match:
                                lib = match.group(1)
                                if lib in ["torch", "tensorflow", "jax", "numpy", "pandas"]:
                                    found_libs.add(lib)
                except Exception:
                    continue
    return list(found_libs)