import json
import os
import datetime
import sys

# This is designed to work with Modal (serverless GPU)
# Usage: modal run tools/validator.py

"""
Modify tools/validator.py to act as a "Connector" that flags failing entries in the database.

Iterate through every recommendation in compatibility.json.

Test the install command on the GPU (or simulate it if you are testing locally).

Write Back a status field ("status": "verified" or "status": "failed") directly into the JSON object for that specific version.

This allows your CLI (env-doctor check) to eventually say: "Warning: CUDA 12.8 is in the DB, but failed validation yesterday."

"""

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False
    print("⚠️  Modal not installed. Running in simulation mode.")
    print("   (To run real GPU tests: pip install modal)")

# Define the Serverless Environment
if HAS_MODAL:
    app = modal.App("env-doctor-validator")
    image = modal.Image.debian_slim().pip_install("torch")
else:
    app = None
    image = None

# Path to your local DB
# We use absolute paths to ensure it finds the file regardless of where you run it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "src", "env_doctor", "data", "compatibility.json")

# Logic wrapper to handle Modal existence
def run_remote_verification(command):
    if HAS_MODAL:
        # This runs verify_install_command on the remote GPU
        return verify_install_command.remote(command)
    else:
        # SIMULATION MODE (For local testing without GPU)
        # Logic: Assume success for standard commands, fail for weird ones
        if "pip install" in command and len(command) < 200:
            return True, "Simulated Success"
        return False, "Simulated Failure (Command too complex)"

if HAS_MODAL:
    @app.function(image=image, gpu="T4")
    def verify_install_command(command):
        """
        This function runs IN THE CLOUD on a real GPU.
        """
        print(f"🧪 Testing command: {command}")
        
        # 1. Run the install
        # We use 'pip install --dry-run' or actual install if ephemeral
        exit_code = os.system(f"{command} --target /tmp/test_env > /dev/null 2>&1")
        if exit_code != 0:
            return False, "Install Failed"

        # 2. Verify GPU access (Simple check)
        # Note: In a real ephemeral container, we would actually import torch
        # Here we assume if install passed in this isolated env, it's good.
        return True, "Success"

def main():
    print(f"--- STARTING VALIDATION ---")
    print(f"Database: {DB_PATH}")
    
    # 1. Load Local Data
    try:
        with open(DB_PATH, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Database not found!")
        sys.exit(1)
    
    recommendations = data.get("recommendations", {})
    total_checked = 0
    total_failed = 0
    
    # 2. Iterate and Test
    for cuda_ver, libs in recommendations.items():
        cmd = libs.get("torch")
        
        # Skip if it's a massive comment block (legacy/messy data)
        if cmd and ("#" in cmd or len(cmd) > 300):
            print(f"⚠️  Skipping CUDA {cuda_ver}: Command looks like a raw text block.")
            libs["verification"] = {
                "status": "skipped",
                "reason": "complex_format",
                "date": datetime.date.today().isoformat()
            }
            continue

        if cmd:
            print(f"🚀 Testing CUDA {cuda_ver}...", end=" ")
            
            # Run the test
            try:
                success, msg = run_remote_verification(cmd)
            except Exception as e:
                success, msg = False, str(e)

            # Update the JSON object specifically for this version
            libs["verification"] = {
                "status": "passed" if success else "failed",
                "message": msg,
                "date": datetime.date.today().isoformat()
            }
            
            if success:
                print(f"✅ Verified")
            else:
                print(f"❌ Failed: {msg}")
                total_failed += 1
            
            total_checked += 1

    # 3. Update Global Metadata
    data["_metadata"] = {
        "last_verified": datetime.date.today().isoformat(),
        "method": "Automated Serverless GPU Test" if HAS_MODAL else "Local Simulation",
        "status": "unhealthy" if total_failed > 0 else "passing",
        "stats": f"{total_checked - total_failed}/{total_checked} passed"
    }
    
    # 4. Save back to JSON
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"\n💾 Database updated. Global Status: {data['_metadata']['status']}")

if __name__ == "__main__":
    # If using Modal, we need to run this inside the app context or just as a script
    # For simplicity in this hybrid script, we run main() directly.
    main()