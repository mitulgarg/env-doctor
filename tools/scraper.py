import requests
from bs4 import BeautifulSoup
import json
import os
import re

DB_PATH = os.path.join("src", "env_doctor", "compatibility.json")
PYTORCH_URL = "https://pytorch.org/get-started/previous-versions/"

CUDA_LINE_RE = re.compile(
    r"pip install .*?(torch|torchvision|torchaudio).*?(cu\d{3,4})",
    re.IGNORECASE
)

def load_existing_db():
    if not os.path.exists(DB_PATH):
        return {"driver_to_cuda": {}, "recommendations": {}}
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)
    print("💾 Saved", DB_PATH)

def parse_cuda_version(token: str) -> str:
    """
    Convert cu118 → 11.8, cu121 → 12.1, cu1210 → 12.10
    """
    digits = token.replace("cu", "")
    if len(digits) == 3:   # 118 → 11.8
        return f"{digits[:2]}.{digits[2]}"
    if len(digits) == 4:   # 1210 → 12.10
        return f"{digits[:2]}.{digits[2:]}"
    return None

def scrape_pytorch_versions(data):
    print("🕷️ Scraping PyTorch page...")

    try:
        response = requests.get(PYTORCH_URL, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print("❌ Failed:", e)
        return data

    soup = BeautifulSoup(response.text, "html.parser")

    updates = 0
    recommendations = data.get("recommendations", {})

    for pre in soup.find_all("pre"):
        lines = pre.get_text().splitlines()

        for line in lines:
            line = line.strip()

            # Skip comments
            if line.startswith("#"):
                continue

            match = CUDA_LINE_RE.search(line)
            if not match:
                continue

            cuda_token = match.group(2)   # cu118
            cuda_version = parse_cuda_version(cuda_token)
            if not cuda_version:
                continue

            # Clean the command
            command = " ".join(line.split())

            if cuda_version not in recommendations:
                recommendations[cuda_version] = {}

            current = recommendations[cuda_version].get("torch")

            # Only update if different
            if command != current:
                recommendations[cuda_version]["torch"] = command

                if "tensorflow" not in recommendations[cuda_version]:
                    recommendations[cuda_version]["tensorflow"] = "Manual verification needed"

                print(f"   ✅ CUDA {cuda_version}: {command}")
                updates += 1

    print(f"✨ Total updates: {updates}")
    data["recommendations"] = recommendations
    return data

def main():
    print("--- STARTING SCRAPER ---")
    data = load_existing_db()
    updated = scrape_pytorch_versions(data)
    save_db(updated)
    print("--- DONE ---")

if __name__ == "__main__":
    main()
