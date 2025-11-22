🩺 Env-Doctor: The AI Environment Fixer

Stop guessing which PyTorch version works with your NVIDIA Driver.

env-doctor is a CLI tool that bridges the gap between your hardware (NVIDIA Drivers) and your software (Python Libraries). It scans your system, detects your GPU driver version, and tells you exactly which pre-compiled binaries (wheels) will work without crashing.

The Problem :-

AI Development relies on a fragile "Tripod of Compatibility":

GPU Driver: (Kernel level, hard to change)

System CUDA: (Compiler level, used for custom builds)

Python Wheels: (Library level, comes bundled with its own CUDA)

If you install a PyTorch wheel bundled with CUDA 12.1, but your Driver only supports CUDA 11.8, your code crashes silently or reverts to CPU mode. pip does not check this for you. We do.

⚡ Installation

git clone [https://github.com/yourusername/env-doctor.git](https://github.com/yourusername/env-doctor.git)
cd env-doctor
pip install -e .


🛠️ Usage

1. Diagnose your Environment

Checks if your currently installed libraries match your physical hardware.

doctor check


Output:

✅ GPU Driver Found: 535.129 (Supports CUDA 12.2)
❌ CRITICAL CONFLICT: Torch requires CUDA 12.4, but Driver maxes at 12.2!

2. Get the Safe Install Command

Don't guess. Get the exact command to install the compatible version.

doctor install torch


Output:

⬇️ Run this command to install the SAFE version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

3. Scan your Project

Automatically detect which AI libraries you are importing in your current directory.

doctor scan


🧩 Architecture

Leg 1 (Hardware): We use nvidia-ml-py to query the live driver version.

Leg 2 (System): We check nvcc for compilation compatibility.

Leg 3 (Software): We map safe versions against a curated compatibility database(Currently Static JSON for POC).

