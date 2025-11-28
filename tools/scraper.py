import requests
from bs4 import BeautifulSoup
import re
import json
import os
import sys

# Path to your DB file
DB_PATH = os.path.join("src", "env_doctor", "compatibility.json")
PYTORCH_URL = "https://pytorch.org/get-started/previous-versions/"

# Regex to find lines that are actual pip commands for CUDA
# Matches: "pip install" ... "cu118" or "+cu118"
CUDA_CMD_RE = re.compile(r"pip\s+install\s+.*torch.*(cu\d{3,4})", re.IGNORECASE)

def load_existing_db():
    if not os.path.exists(DB_PATH):
        print(f"⚠️  Could not find {DB_PATH}. Creating new structure.")
        return {"driver_to_cuda": {}, "recommendations": {}}
    
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)
    print(f"💾  Saved updates to {DB_PATH}")

def parse_cuda_version(token: str) -> str:
    """
    Convert 'cu118' -> '11.8', 'cu121' -> '12.1'
    """
    digits = re.sub(r"[^0-9]", "", token) # Remove 'cu' or '+'
    
    if len(digits) == 3:   # 118 -> 11.8
        return f"{digits[:2]}.{digits[2]}"
    if len(digits) >= 4:   # 1210 -> 12.1
        return f"{digits[:2]}.{digits[2]}"
    return None

def scrape_pytorch_versions(data):
    print(f"🕸️  Scraping {PYTORCH_URL}...")
    
    try:
        response = requests.get(PYTORCH_URL, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch page: {e}")
        return data

    soup = BeautifulSoup(response.text, 'html.parser')
    recommendations = data.get("recommendations", {})
    updates_count = 0

    # 1. Find all code blocks
    for pre in soup.find_all("pre"):
        # CRITICAL FIX: Use separator='\n' to handle <br> tags correctly
        content = pre.get_text(separator="\n")
        lines = content.splitlines()

        for line in lines:
            clean_line = line.strip()

            # Skip comments, empty lines, or ROCm/CPU commands
            if not clean_line or clean_line.startswith("#"):
                continue
            if "rocm" in clean_line.lower() or "cpu" in clean_line.lower():
                continue
            if "pip install" not in clean_line:
                continue

            # Match CUDA commands
            match = CUDA_CMD_RE.search(clean_line)
            if match:
                cuda_token = match.group(1) # e.g. "cu118"
                cuda_ver = parse_cuda_version(cuda_token)
                
                if not cuda_ver:
                    continue

                # CRITICAL LOGIC: First Come, First Served.
                # The page lists newest versions first. If we already have an entry
                # for "11.8", it is likely the newest one we found earlier.
                # We DO NOT overwrite it with older versions found further down.
                if cuda_ver not in recommendations:
                    
                    # Clean up the command (remove multiple spaces)
                    final_cmd = " ".join(clean_line.split())
                    
                    recommendations[cuda_ver] = {
                        "torch": final_cmd,
                        "tensorflow": "Manual verification needed" 
                    }
                    
                    print(f"   ✅ NEW: CUDA {cuda_ver} -> {final_cmd[:60]}...")
                    updates_count += 1

    data["recommendations"] = recommendations
    print(f"   ✨ Total new entries added: {updates_count}")
    return data

def main():
    print("--- STARTING AUTO-UPDATER ---")
    data = load_existing_db()
    
    # Run Scraper
    updated_data = scrape_pytorch_versions(data)
    
    # Save result
    save_db(updated_data)
    print("--- DONE ---")

if __name__ == "__main__":
    main()