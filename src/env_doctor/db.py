"""
The 'Golden Truth' Database.
In a production version, this would fetch a JSON from a GitHub Pages URL.
For this MVP, we hardcode the known critical compatibility layers.
"""

# This maps NVIDIA Driver Versions to the MAX supported CUDA Toolkit version.
# Source: NVIDIA Compatibility Documentation
# Format: "Driver Major Version": "Max Supported CUDA"
DRIVER_TO_CUDA = {
    # Linux Driver Branches
    "550": "12.4",
    "545": "12.3",
    "535": "12.2", # LTS
    "530": "12.1",
    "525": "12.0", # LTS
    "520": "11.8",
    "515": "11.7",
    "510": "11.6",
    "470": "11.4", # Very common LTS
    "465": "11.3",
    "460": "11.2",
    "455": "11.1",
    "450": "11.0",
}

# Safe 'Wheel' recommendations based on the Max Supported CUDA.
# This maps a 'Max CUDA' (from above) to specific install commands.
RECOMMENDATIONS = {
    "12.4": {
        "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "tensorflow": "pip install tensorflow==2.16.1  # (TensorFlow often relies on system CUDA > 2.10)"
    },
    "12.1": {
        "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
        "tensorflow": "pip install tensorflow==2.15.0"
    },
    "11.8": {
        "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "tensorflow": "pip install tensorflow==2.14.0"
    },
    "11.7": {
        "torch": "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117",
        "tensorflow": "pip install tensorflow==2.13.0"
    },
    # Fallback for very old drivers
    "11.x": {
        "torch": "pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu116",
        "tensorflow": "pip install tensorflow==2.11.0"
    }
}

def get_max_cuda_for_driver(driver_version: str) -> str:
    """
    Given a driver version string (e.g., '535.129.03'), returns the max supported CUDA (e.g., '12.2').
    """
    try:
        major_version = driver_version.split('.')[0]
        
        # Exact match
        if major_version in DRIVER_TO_CUDA:
            return DRIVER_TO_CUDA[major_version]
        
        # If not exact, find the closest lower bound (conservative approach)
        # This is simple logic; a real DB would be more robust.
        available_drivers = sorted([int(x) for x in DRIVER_TO_CUDA.keys()], reverse=True)
        driver_int = int(major_version)
        
        for known_driver in available_drivers:
            if driver_int >= known_driver:
                return DRIVER_TO_CUDA[str(known_driver)]
        
        return "10.0" # Extremely safe fallback
    except Exception:
        return "Unknown"

def get_install_command(library: str, max_cuda: str) -> str:
    """Returns the install string for a library given the system constraint."""
    
    # 1. Try exact match
    if max_cuda in RECOMMENDATIONS:
        return RECOMMENDATIONS[max_cuda].get(library, "No data for this library")
    
    # 2. Logic for finding 'closest' if exact CUDA version isn't in our specific map
    # (Simplified for MVP: Downgrade to 11.8 if between 11.8 and 12.1)
    try:
        cuda_float = float(max_cuda)
        if cuda_float >= 12.1:
            return RECOMMENDATIONS["12.1"].get(library)
        elif cuda_float >= 11.8:
            return RECOMMENDATIONS["11.8"].get(library)
        else:
            return RECOMMENDATIONS["11.7"].get(library)
    except:
        return "Could not determine safe version."