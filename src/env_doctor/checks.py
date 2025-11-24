import sys
import shutil
import subprocess
import importlib
import os
import re
import json

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
        return None

def get_system_cuda_version():
    """
    Leg 2: System Compiler Check.
    Checks `nvcc --version`. Returns string (e.g., '11.8') or None.
    """
    nvcc_path = shutil.which("nvcc")
    
    if not nvcc_path and os.path.exists("/usr/local/cuda/bin/nvcc"):
        nvcc_path = "/usr/local/cuda/bin/nvcc"
            
    if not nvcc_path:
        return None

    try:
        result = subprocess.check_output([nvcc_path, "--version"], encoding="utf-8")
        match = re.search(r"release (\d+\.\d+)", result)
        if match:
            return match.group(1)
    except Exception:
        return None
    return None

def get_installed_library_version(lib_name):
    """
    Leg 3: Python Library Check.
    Returns dictionary with 'version', 'cuda', and 'cudnn' (if detected).
    """
    try:
        lib = importlib.import_module(lib_name)
        version = getattr(lib, "__version__", "Unknown")
        cuda_ver = "Unknown"
        cudnn_ver = "Unknown"
        
        if lib_name == "torch":
            try:
                cuda_ver = lib.version.cuda
                if hasattr(lib.backends, 'cudnn'):
                    raw_cudnn = lib.backends.cudnn.version()
                    if raw_cudnn:
                        major = raw_cudnn // 1000
                        minor = (raw_cudnn % 1000) // 100
                        patch = raw_cudnn % 100
                        cudnn_ver = f"{major}.{minor}.{patch}"
            except AttributeError:
                pass
        
        elif lib_name == "tensorflow":
            try:
                sys_config = getattr(lib, "sysconfig", None)
                if sys_config and hasattr(sys_config, "get_build_info"):
                    build_info = sys_config.get_build_info()
                    cuda_ver = build_info.get("cuda_version", "Unknown")
                    cudnn_ver = build_info.get("cudnn_version", "Unknown")
            except Exception:
                pass

        return {"version": version, "cuda": cuda_ver, "cudnn": cudnn_ver}
    except ImportError:
        return None

def scan_imports_in_folder(folder_path="."):
    """
    Scans all .py files in the directory to find 'import torch', 'import tensorflow', etc.
    """
    found_libs = set()
    import_regex = re.compile(r"^\s*(?:import|from)\s+(\w+)")
    
    for root, dirs, files in os.walk(folder_path):
        if "venv" in root or ".git" in root: 
            continue 

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

def load_migrations():
    """Loads the migration rules from migrations.json"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, "migrations.json")
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def check_broken_imports():
    """
    Scans code for imports that are dead/moved in the currently installed versions.
    """
    print("\n🦜 CODE MIGRATION CHECK (LangChain / Pydantic / OpenAI)")
    
    migration_db = load_migrations()
    issues_found = 0

    # 1. Loop through every library in our DB
    for lib_name, config in migration_db.items():
        
        # Check if user has the library installed
        installed = get_installed_library_version(lib_name)
        if not installed:
            continue

        # Check if installed version is NEWER than the trigger
        try:
            installed_major = int(installed['version'].split('.')[0])
            trigger_major = int(config['trigger_version'].split('.')[0])
        except (ValueError, IndexError):
            continue # Version parsing failed
        
        if installed_major < trigger_major:
            continue # They are on the old version, so old imports are valid.

        print(f"    Analyzing {lib_name} (v{installed['version']}) usage...")

        # 2. Scan files for forbidden strings
        for root, dirs, files in os.walk("."):
            if "venv" in root or ".git" in root: continue 
            
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f):
                                for old_str, rule in config['rules'].items():
                                    if old_str in line:
                                        print(f"    ❌ DEPRECATED in {file}:{i+1}")
                                        print(f"       Found: '{old_str}'")
                                        print(f"       Moved to: '{rule['new_path']}'")
                                        print(f"       Action: {rule.get('fix_cmd', 'Update code manually')}")
                                        issues_found += 1
                    except Exception:
                        pass

    if issues_found == 0:
        print("    ✅ No deprecated imports detected.")
    else:
        print(f"\n    ⚠️  Found {issues_found} migration issues.")