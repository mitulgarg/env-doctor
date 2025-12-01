import sys
import shutil
import subprocess
import importlib
import importlib.metadata
import os
import re
import json


try:
    from nvidia import nvidia_smi       #new nvidia lib
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def get_nvidia_driver_version():
    # 1. Try NVML
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion().decode()
        pynvml.nvmlShutdown()
        return driver
    except:
        pass

    # 2. Try nvidia-smi
    try:
        out = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
        match = re.search(r"Driver Version:\s+(\d+\.\d+)", out)
        if match:
            return match.group(1)
    except:
        pass

    # 3. Fail
    return None

def get_system_cuda_version():
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path and os.path.exists("/usr/local/cuda/bin/nvcc"):
        nvcc_path = "/usr/local/cuda/bin/nvcc"
    if not nvcc_path: return None

    try:
        result = subprocess.check_output([nvcc_path, "--version"], encoding="utf-8")
        match = re.search(r"release (\d+\.\d+)", result)
        if match: return match.group(1)
    except Exception:
        return None
    return None

def get_installed_library_version(lib_name):
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
                        # PyTorch stores as int (8700 -> 8.7.0)
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

        elif lib_name == "jax":
            try:
                import jaxlib
                if hasattr(jaxlib, "version"):
                    # Try to infer from installed nvidia packages
                    packages = [d.metadata['Name'] for d in importlib.metadata.distributions()]
                    if any("nvidia-cuda-runtime-cu12" in p for p in packages):
                        cuda_ver = "12.x (via pip)"
                    elif any("nvidia-cuda-runtime-cu11" in p for p in packages):
                        cuda_ver = "11.x (via pip)"
            except ImportError:
                cuda_ver = "CPU Only (jaxlib not found)"

        return {"version": version, "cuda": cuda_ver, "cudnn": cudnn_ver}
    except ImportError:
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