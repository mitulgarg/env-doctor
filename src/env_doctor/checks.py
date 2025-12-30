# Keep for backward compatibility (thin wrappers)

import sys
import shutil
import subprocess
import importlib
import importlib.metadata
import os
import re
import json
from env_doctor.core.registry import DetectorRegistry

try:
    from nvidia import nvidia_smi       #new nvidia lib
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# Backward compatibility wrappers
def get_nvidia_driver_version():
    """Legacy wrapper - use NvidiaDriverDetector directly."""
    detector = DetectorRegistry.get("nvidia_driver")
    result = detector.detect()
    return result.version if result.detected else None

def get_system_cuda_version():
    """Legacy wrapper - use CudaToolkitDetector directly."""
    detector = DetectorRegistry.get("cuda_toolkit")
    result = detector.detect()
    return result.version if result.detected else None
def get_installed_library_version(lib_name):
    """Legacy wrapper - use PythonLibraryDetector directly."""
    from env_doctor.detectors.python_libraries import PythonLibraryDetector
    detector = PythonLibraryDetector(lib_name)
    result = detector.detect()
    if result.detected:
        return {
            "version": result.version,
            "cuda": result.metadata.get("cuda_version", "Unknown"),
            "cudnn": result.metadata.get("cudnn_version", "Unknown")
        }
    return None

def scan_imports_in_folder(folder_path="."):
    found_libs = set()
    import_regex = re.compile(r"^\s*(?:import|from)\s+(\w+)")
    
    for root, dirs, files in os.walk(folder_path):
        if "venv" in root or ".git" in root: continue 

        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            match = import_regex.match(line)
                            if match:
                                lib = match.group(1)
                                if lib in ["torch", "tensorflow", "jax", "flax", "numpy", "pandas"]:
                                    found_libs.add(lib)
                except Exception:
                    continue
    return list(found_libs)

def load_migrations():
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, "migrations.json")
    try:
        with open(json_path, "r") as f: return json.load(f)
    except FileNotFoundError: return {}

def check_broken_imports():
    print("\n🦜 CODE MIGRATION CHECK (LangChain / Pydantic / OpenAI)")
    migration_db = load_migrations()
    issues_found = 0

    for lib_name, config in migration_db.items():
        installed = get_installed_library_version(lib_name)
        if not installed: continue

        try:
            installed_major = int(installed['version'].split('.')[0])
            trigger_major = int(config['trigger_version'].split('.')[0])
        except (ValueError, IndexError): continue
        
        if installed_major < trigger_major: continue

        print(f"    Analyzing {lib_name} (v{installed['version']}) usage...")

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
                    except Exception: pass

    if issues_found == 0:
        print("    ✅ No deprecated imports detected.")
    else:
        print(f"\n    ⚠️  Found {issues_found} migration issues.")