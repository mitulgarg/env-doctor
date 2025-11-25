🩺 Env-Doctor: The AI Environment Fixer
Stop guessing which PyTorch version works with your NVIDIA driver.
Env-Doctor is a CLI tool that bridges the gap between your hardware (NVIDIA drivers) and your software (Python AI libraries). It scans your system, detects your GPU driver version, and tells you exactly which pre-compiled binaries (wheels) will work — without random crashes or cryptic CUDA errors.

🛡️ Verified Daily — A Self-Improving Database
We don’t guess compatibility. Env-Doctor is powered by an automated verification system:


Scraper – Watches PyTorch & NVIDIA release notes every 24 hours


Validator – Physically tests new versions on serverless GPUs (T4/A100)


Hybrid Cache – Your CLI fetches the latest compatibility rules from GitHub, and falls back to local data when offline



🔴 The Problem: The Tripod of Compatibility
AI development depends on three layers that must align perfectly.
If any one is mismatched, you get silent failures or cryptic C++ CUDA errors.
Leg 1 — GPU Driver (Kernel Level)


Hard to change


Determines your maximum supported CUDA version


Leg 2 — System CUDA Toolkit (Compiler Level)


Used only when building from source (e.g., Flash-Attention, xFormers)


Must match the library you’re compiling


Leg 3 — Python Wheels (Library Level)


Wheels bundle their own CUDA runtime


If this > Driver’s max CUDA → Crash


Env-Doctor checks all three legs and ensures they stand together.

⚡ Installation
From PyPI (Recommended)
(Not yet published — coming soon)
pip install env-doctor

From Source (Development)
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor
pip install -e .


🛠️ Usage
1️⃣ Diagnose Your Environment
Checks hardware, system paths, and installed libraries for conflicts.
doctor check

Sample Output
🩺 ENV-DOCTOR DIAGNOSIS
==============================
🛡️  DB Verified: 2025-11-24 (Automated Serverless GPU Test)

✅  GPU Driver Found: 535.129 (Supports CUDA 12.2)
✅  System CUDA (nvcc): 11.8

📦 Found torch: v2.2.1
   → Bundled CUDA: 12.1
   ✅ Compatible with Driver.

🏭 COMPILATION HEALTH (Flash-Attention / AutoGPTQ)
❌ ASYMMETRY DETECTED:
   System (11.8) != Torch (12.1)
   → pip install flash-attention will FAIL.

🦜 CODE MIGRATION CHECK
❌ Deprecated in src/main.py:4
   Found: 'langchain.chat_models'
   Moved to: 'langchain_community.chat_models'


2️⃣ Get the Safe Install Command
Stop guessing which torch/cuXX wheel works on your machine.
doctor install torch

Output
⬇️ Run this command to install the SAFE version:
---------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
---------------------------------------------------


3️⃣ Scan Your Project for AI Libraries
Automatically detects imports and suggests fixes for deprecated or incompatible APIs.
doctor scan


🧩 Architecture
🧠 The Brain (Data)


compatibility.json
Maps GPU driver → max CUDA → compatible wheels


migrations.json
Maps deprecated API imports to correct replacements (e.g., LangChain v0.2+)


✋ The Hands (CLI)


checks.py — Detects driver, system CUDA, torch wheels (via NVML & nvcc)


db.py — Hybrid online/offline compatibility loader


⚙️ The Updater (CI/CD Automation)


tools/scraper.py — Fetches new releases from NVIDIA + PyTorch


tools/validator.py — Spins up cloud GPUs to verify compatibility before updates are accepted



📄 License
MIT

If you'd like, I can also generate:
📌 badges (PyPI, version, downloads, CI status)
📌 a clean project banner image
📌 a pypi.org-optimized README variant
📌 installable CLI help (doctor --help) section
📌 improved architecture diagram (ASCII or image)
Just say "add badges" or "make this PyPI-ready" or "generate banner."