🩺 Env-Doctor: The AI Environment Fixer
Stop guessing which PyTorch version works with your NVIDIA Driver.
env-doctor is a CLI tool that bridges the gap between your hardware (NVIDIA Drivers) and your software (Python Libraries). It scans your system, detects your GPU driver version, and tells you exactly which pre-compiled binaries (wheels) will work without crashing.
🛡️ Verified Daily
We don't just guess compatibility. This tool is powered by a Self-Improving Database.
Scraper: A bot watches PyTorch and NVIDIA release notes every 24 hours.
Validator: New versions are physically tested on Serverless GPUs (T4/A100s) to ensure they actually run.
Hybrid Cache: Your CLI automatically fetches the latest verified rules from the cloud, but falls back to a local cache if you are offline.
🔴 The Problem: "The Tripod of Compatibility"
AI Development relies on three layers that must align perfectly. If they don't, you get silent failures or cryptic C++ errors.
Leg 1: GPU Driver (Kernel Level)
Constraint: Hard to change. Sets the "Speed Limit" (Max CUDA version).
Leg 2: System CUDA (Compiler Level)
Constraint: Used only when building from source (e.g., pip install flash-attention). Must match the library.
Leg 3: Python Wheels (Library Level)
Constraint: Comes bundled with its own CUDA runtime. If this is newer than Leg 1, CRASH.
env-doctor checks all three legs and ensures they stand together.

⚡ Installation (not yet, will be configured soon)
From PyPI (Recommended)
pip install env-doctor


From Source (For Development)
git clone [https://github.com/mitulgarg/env-doctor.git](https://github.com/mitulgarg/env-doctor.git)
cd env-doctor
pip install -e .


🛠️ Usage
1. Diagnose your Environment
Checks hardware, system paths, and installed libraries for conflicts.
doctor check


Sample Output:
🩺  ENV-DOCTOR DIAGNOSIS  🩺
==============================
🛡️  DB Verified: 2025-11-24 (Automated Serverless GPU Test)
✅  GPU Driver Found: 535.129 (Supports CUDA 12.2)
✅  System CUDA (nvcc): 11.8
------------------------------
📦  Found torch: v2.2.1
    -> Bundled CUDA: 12.1
    ✅ Compatible with Driver.

🏭  COMPILATION HEALTH (For Flash-Attention/AutoGPTQ)
❌  ASYMMETRY DETECTED: System (11.8) != Torch (12.1)
    -> pip install flash-attention will FAIL.

🦜 CODE MIGRATION CHECK
❌  DEPRECATED in src/main.py:4
    Found: 'langchain.chat_models'
    Moved to: 'langchain_community.chat_models'


2. Get the Safe Install Command
Don't guess. Get the exact command to install the compatible version for your specific machine.
doctor install torch


Output:
⬇️   Run this command to install the SAFE version:
---------------------------------------------------
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
---------------------------------------------------


3. Scan your Project
Automatically detect which AI libraries you are importing in your current directory and suggest fixes.
doctor scan


🧩 Architecture
The Brain (Data)
compatibility.json: A curated matrix mapping Driver Versions -> Max CUDA -> Compatible Wheels.
migrations.json: A dictionary of breaking API changes (e.g., LangChain v0.2 imports) to help you migrate old code.
The Hands (CLI)
checks.py: Uses nvidia-ml-py (NVML) to query the live driver version and nvcc for system checks.
db.py: Implements a "Hybrid Loader" that fetches the latest JSON from GitHub if internet is available, ensuring you always have the latest compatibility data without upgrading the package.
The Updater (CI/CD)
tools/scraper.py: Runs on GitHub Actions to find new releases.
tools/validator.py: Spins up a cloud GPU to verify installation commands before pushing them to the database.
License
MIT
