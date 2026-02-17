import json
import os
import sys
import time
import requests


def _safe_print(msg: str, **kwargs) -> None:
    """Print with fallback for encoding issues (e.g., emojis on Windows cp1252)."""
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        # Strip emojis and non-ASCII characters for terminals that can't handle them
        safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
        print(safe_msg, **kwargs)


# 1. The URL where your Scraper pushes the latest data

REMOTE_URL = "https://raw.githubusercontent.com/mitulgarg/env-doctor/main/src/env_doctor/data/compatibility.json"

# 2. Local Cache File (in user's home dir) so we don't hit GitHub every single run
CACHE_FILE = os.path.expanduser("~/.env_doctor_cache.json")

# 3. How old can the cache be before we check online? (24 hours)
CACHE_TTL = 24 * 60 * 60 

def load_bundled_json():
    """Loads the 'Factory Default' JSON shipped with the package."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, "data", "compatibility.json")
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"driver_to_cuda": {}, "recommendations": {}}

def load_database():
    """
    Hybrid Loader:
    1. Check if we have a fresh local cache.
    2. If not, try to download latest from GitHub.
    3. If download fails (offline), fall back to Bundled JSON.
    """
    
    # Step A: Is Cache Fresh?
    if os.path.exists(CACHE_FILE):
        try:
            last_modified = os.path.getmtime(CACHE_FILE)
            if time.time() - last_modified < CACHE_TTL:
                # Cache is fresh, use it
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
        except:
            pass # Cache corrupted, ignore it

    # Step B: Try Fetching Remote
    # We use print with end="" to show status without cluttering if it's fast
    _safe_print("Checking for latest compatibility data...", end=" ", flush=True)
    try:
        response = requests.get(REMOTE_URL, timeout=1.5) # Short timeout so CLI feels snappy
        if response.status_code == 200:
            data = response.json()
            # Save to cache
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f)
            _safe_print("Updated.")
            return data
        else:
            _safe_print("Server error. Using local DB.")
    except requests.RequestException:
        _safe_print("Offline. Using local DB.")

    # Step C: Fallback to Bundled
    return load_bundled_json()

# --- Load Data ---
DB_DATA = load_database()
DRIVER_TO_CUDA = DB_DATA.get("driver_to_cuda", {})
RECOMMENDATIONS = DB_DATA.get("recommendations", {})

def get_max_cuda_for_driver(driver_version: str) -> str:
    """
    Given a driver version string (e.g., '535.129.03'), returns the max supported CUDA (e.g., '12.2').
    """
    try:
        major_version = driver_version.split('.')[0]
        
        # Exact match
        if major_version in DRIVER_TO_CUDA:
            return DRIVER_TO_CUDA[major_version]
        
        # Closest Lower Bound logic
        # Convert keys to ints for numerical comparison
        available_drivers = sorted([int(x) for x in DRIVER_TO_CUDA.keys()], reverse=True)
        driver_int = int(major_version)
        
        for known_driver in available_drivers:
            if driver_int >= known_driver:
                return DRIVER_TO_CUDA[str(known_driver)]
        
        return "10.0" # Safe fallback
    except Exception:
        return "Unknown"

def get_install_command(library: str, max_cuda: str) -> str:
    """
    
    Returns the install string for a library given the system constraint. 
    Checks nvcc and pytorch CUDA match for flash-attn
    
    """
    # flash-attention



    # Exact match in DB
    if max_cuda in RECOMMENDATIONS:
        val = RECOMMENDATIONS[max_cuda].get(library)
        if val: return val
    
    # Logic for finding 'closest' if exact CUDA version isn't in our specific map
    try:
        cuda_float = float(max_cuda)
        if cuda_float >= 12.1:
            return RECOMMENDATIONS.get("12.1", {}).get(library, "Unknown")
        elif cuda_float >= 11.8:
            return RECOMMENDATIONS.get("11.8", {}).get(library, "Unknown")
        else:
            return RECOMMENDATIONS.get("11.7", {}).get(library, "Unknown")
    except:
        return "Could not determine safe version."


# --- CUDA Toolkit Install Data ---

def load_cuda_toolkit_install_data():
    """Load CUDA Toolkit installation instructions from JSON."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, "data", "cuda_toolkit_install.json")
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"cuda_versions": {}, "version_recommendation_map": {}}


CUDA_INSTALL_DATA = load_cuda_toolkit_install_data()


# --- Python Compatibility Data ---

def load_python_compatibility():
    """Load Python version compatibility data from JSON."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, "data", "python_compatibility.json")
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"python_version_constraints": [], "dependency_cascades": []}


PYTHON_COMPAT_DATA = load_python_compatibility()


def get_recommended_cuda_toolkit(max_cuda: str):
    """
    Given the driver's max CUDA version, return the recommended
    CUDA Toolkit version to install.

    Args:
        max_cuda: Maximum CUDA version supported by driver (e.g., "12.6")

    Returns:
        Recommended toolkit version string (e.g., "12.6"), or None
    """
    version_map = CUDA_INSTALL_DATA.get("version_recommendation_map", {})

    # Exact match
    if max_cuda in version_map:
        return version_map[max_cuda]

    # Closest lower bound (same logic as driver_to_cuda)
    try:
        cuda_float = float(max_cuda)
        available = sorted(
            [(float(k), v) for k, v in version_map.items()],
            reverse=True
        )
        for version, toolkit in available:
            if cuda_float >= version:
                return toolkit
    except ValueError:
        pass

    return None


def get_cuda_install_steps(toolkit_version: str, platform_keys: list):
    """
    Get installation steps for a specific CUDA Toolkit version and platform.

    Args:
        toolkit_version: CUDA Toolkit version (e.g., "12.6")
        platform_keys: List of platform keys to try, in priority order

    Returns:
        Dict with installation steps, or None if not found
    """
    versions = CUDA_INSTALL_DATA.get("cuda_versions", {})
    version_data = versions.get(toolkit_version)

    if not version_data:
        return None

    platforms = version_data.get("platforms", {})

    # Try each platform key in order
    for key in platform_keys:
        if key in platforms:
            result = platforms[key].copy()
            result["cuda_version"] = toolkit_version
            result["display_name"] = version_data.get("display_name", f"CUDA Toolkit {toolkit_version}")
            result["download_page"] = version_data.get("download_page", "")
            return result

    return None