import json
import os
import time
import requests

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
    print("🌐 Checking for latest compatibility data...", end=" ", flush=True)
    try:
        response = requests.get(REMOTE_URL, timeout=1.5) # Short timeout so CLI feels snappy
        if response.status_code == 200:
            data = response.json()
            # Save to cache
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f)
            print("Updated. ✅")
            return data
        else:
            print("Server error. Using local DB. ⚠️")
    except requests.RequestException:
        print("Offline. Using local DB. 🔌")

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
    """Returns the install string for a library given the system constraint."""
    
    # 1. Exact match in DB
    if max_cuda in RECOMMENDATIONS:
        val = RECOMMENDATIONS[max_cuda].get(library)
        if val: return val
    
    # 2. Logic for finding 'closest' if exact CUDA version isn't in our specific map
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