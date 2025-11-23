import json
import os

def load_database():
    """Loads the compatibility matrix from the JSON file located in the package."""
    # Find compatibility.json relative to this file
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, "compatibility.json")
    
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback if file is missing (should not happen in proper install)
        return {"driver_to_cuda": {}, "recommendations": {}}

# Load data once when module is imported
DB_DATA = load_database()
DRIVER_TO_CUDA = DB_DATA.get("driver_to_cuda", {})
RECOMMENDATIONS = DB_DATA.get("recommendations", {})

def get_max_cuda_for_driver(driver_version: str) -> str:
    """
    Given a driver version string (e.g., '535.129.03'), returns the max supported CUDA (e.g., '12.2').
    """
    try:
        major_version = driver_version.split('.')[0]
        
        # 1. Exact match check
        if major_version in DRIVER_TO_CUDA:
            return DRIVER_TO_CUDA[major_version]
        
        # 2. Closest Lower Bound logic
        # Convert keys to ints for numerical comparison
        available_drivers = sorted([int(x) for x in DRIVER_TO_CUDA.keys()], reverse=True)
        driver_int = int(major_version)
        
        for known_driver in available_drivers:
            if driver_int >= known_driver:
                return DRIVER_TO_CUDA[str(known_driver)]
        
        return "10.0" # Safe fallback for very old drivers
    except Exception:
        return "Unknown"

def get_install_command(library: str, max_cuda: str) -> str:
    """Returns the install string for a library given the system constraint."""
    
    # 1. Exact match in DB
    if max_cuda in RECOMMENDATIONS:
        val = RECOMMENDATIONS[max_cuda].get(library)
        if val: return val
    
    # 2. Logic for finding 'closest' version if exact CUDA version isn't in our specific map
    # (e.g., Driver supports 12.0, but we only have wheels for 11.8 and 12.1)
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