import requests
from bs4 import BeautifulSoup
import re
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "src", "env_doctor", "data", "compatibility.json")

PYTORCH_URL = "https://pytorch.org/get-started/previous-versions/"
JAX_PYPI_URL = "https://pypi.org/pypi/jax/json"
CUDA_CMD_RE = re.compile(r"pip\s+install\s+.*torch.*(cu\d{3,4})", re.IGNORECASE)

def load_existing_db():
    if not os.path.exists(DB_PATH):
        return {"driver_to_cuda": {}, "recommendations": {}}
    with open(DB_PATH, "r") as f: return json.load(f)

def save_db(data):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f: json.dump(data, f, indent=4)
    print(f"💾  Saved updates to {DB_PATH}")

def parse_cuda_version(token: str) -> str:
    digits = re.sub(r"[^0-9]", "", token)
    if len(digits) == 3: return f"{digits[:2]}.{digits[2]}"
    if len(digits) >= 4: return f"{digits[:2]}.{digits[2]}"
    return None

def scrape_jax_versions(data):
    print(f"🕸️  Scraping JAX (PyPI)...")
    try:
        response = requests.get(JAX_PYPI_URL, timeout=10)
        response.raise_for_status()
        jax_data = response.json()
        latest_version = jax_data["info"]["version"]
        print(f"   ℹ️  Latest JAX Version: {latest_version}")
        
        recommendations = data.get("recommendations", {})
        jax_cmd = f'pip install -U "jax[cuda12_pip]=={latest_version}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
        
        # Update CUDA 12.x keys for JAX
        for key in ["12.1", "12.2", "12.3", "12.4", "12.6"]:
            if key not in recommendations: recommendations[key] = {}
            recommendations[key]["jax"] = jax_cmd
        print(f"   ✅ Updated JAX for CUDA 12.x")
    except Exception as e:
        print(f"❌ Failed to scrape JAX: {e}")
    return data

def scrape_pytorch_versions(data):
    print(f"🕸️  Scraping {PYTORCH_URL}...")
    try:
        response = requests.get(PYTORCH_URL, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed: {e}")
        return data

    soup = BeautifulSoup(response.text, 'html.parser')
    recommendations = data.get("recommendations", {})
    updates = 0

    for pre in soup.find_all("pre"):
        content = pre.get_text(separator="\n")
        lines = content.splitlines()

        for line in lines:
            clean = line.strip()
            if not clean or clean.startswith("#"): continue
            if "rocm" in clean.lower() or "cpu" in clean.lower(): continue
            if "pip install" not in clean: continue

            match = CUDA_CMD_RE.search(clean)
            if match:
                cuda_token = match.group(1)
                cuda_ver = parse_cuda_version(cuda_token)
                if not cuda_ver: continue

                if cuda_ver not in recommendations:
                    final_cmd = " ".join(clean.split())
                    recommendations[cuda_ver] = {
                        "torch": final_cmd,
                        "tensorflow": "Manual verification needed" 
                    }
                    print(f"   ✅ NEW: CUDA {cuda_ver} -> {final_cmd[:30]}...")
                    updates += 1

    data["recommendations"] = recommendations
    print(f"   ✨ PyTorch updates: {updates}")
    return data

def main():
    print(f"--- STARTING AUTO-UPDATER ---")
    data = load_existing_db()
    data = scrape_pytorch_versions(data)
    data = scrape_jax_versions(data)
    save_db(data)
    print("--- DONE ---")

if __name__ == "__main__":
    main()