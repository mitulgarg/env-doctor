import requests
from bs4 import BeautifulSoup
import re
import json
import os
import sys

# Target paths relative to script execution
# Assumes running from root: python tools/scraper.py
DB_PATH = os.path.join("src", "env_doctor", "compatibility.json")
PYTORCH_URL = "https://pytorch.org/get-started/previous-versions/"

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

def scrape_pytorch_versions(current_data):
    print(f"🕸️  Scraping {PYTORCH_URL}...")
    
    try:
        response = requests.get(PYTORCH_URL, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch page: {e}")
        return current_data

    soup = BeautifulSoup(response.text, 'html.parser')
    code_blocks = soup.find_all("pre")
    
    recommendations = current_data.get("recommendations", {})
    updates_count = 0

    for block in code_blocks:
        text = block.get_text()
        
        if "pip install" in text and "torch" in text:
            # Extract CUDA version (e.g. cu118)
            cuda_match = re.search(r'cu(\d+)', text)
            # Extract Torch version (e.g. 2.1.0)
            torch_match = re.search(r'torch==(\d+\.\d+\.\d+)', text)

            if cuda_match and torch_match:
                raw_cuda = cuda_match.group(1) # e.g. "118"
                
                # Format CUDA version to "11.8" or "12.1"
                if len(raw_cuda) == 3:
                    formatted_cuda = f"{raw_cuda[0:2]}.{raw_cuda[2]}"
                elif len(raw_cuda) >= 4: # Handle cases like 1210
                     formatted_cuda = f"{raw_cuda[0:2]}.{raw_cuda[2]}"
                else:
                    continue

                # Clean the command string
                clean_command = text.strip().replace("\n", " ")
                
                # Initialize key if missing
                if formatted_cuda not in recommendations:
                    recommendations[formatted_cuda] = {}

                # Update logic: overwrite if torch version is missing or different
                # (Simple logic: assumes scraper finds newest versions first or last)
                current_torch_cmd = recommendations[formatted_cuda].get("torch", "")
                
                if current_torch_cmd != clean_command:
                    recommendations[formatted_cuda]["torch"] = clean_command
                    
                    # Preserve TensorFlow if it exists so we don't delete manual entries
                    if "tensorflow" not in recommendations[formatted_cuda]:
                        recommendations[formatted_cuda]["tensorflow"] = "Manual verification needed"
                    
                    print(f"   ✅ Updated: CUDA {formatted_cuda} -> Torch {torch_match.group(1)}")
                    updates_count += 1

    current_data["recommendations"] = recommendations
    print(f"   ✨ Total updates found: {updates_count}")
    return current_data

def main():
    print("--- STARTING AUTO-UPDATER ---")
    data = load_existing_db()
    updated_data = scrape_pytorch_versions(data)
    save_db(updated_data)
    print("--- DONE ---")

if __name__ == "__main__":
    main()