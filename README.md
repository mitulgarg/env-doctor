# 🩺 Env-Doctor: The AI Environment Fixer

[![PyPI version](https://badge.fury.io/py/env-doctor.svg)](https://badge.fury.io/py/env-doctor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Stop guessing which PyTorch version works with your NVIDIA driver.**

**Env-Doctor** is a CLI tool that bridges the gap between your hardware (NVIDIA drivers) and your software (Python AI libraries). It scans your system, detects your GPU driver version, and tells you exactly which pre-compiled binaries (wheels) will work — preventing random crashes and cryptic CUDA errors.

---

## 🚀 Key Features

- **🛡️ Automated Verification**: Powered by a database verified daily against real serverless GPUs (T4/A100).
- **🩺 Deep Diagnosis**: Checks the "Tripod of Compatibility":
    1.  **GPU Driver** (Kernel Level)
    2.  **System CUDA** (Compiler Level)
    3.  **Python Wheels** (Library Level)
- **💊 Precise Prescriptions**: Generates the *exact* `pip install` command for your specific driver/CUDA combo.
- **🦜 Migration Assistant**: Scans your code for deprecated imports (e.g., LangChain, OpenAI) and suggests fixes.

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install env-doctor
```

### From Source (Development)
```bash
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor
pip install -e .
```

## 🛠️ Usage

### 1. Diagnose Your Environment
Check hardware, system paths, and installed libraries for conflicts.

```console
$ doctor check

🩺 ENV-DOCTOR DIAGNOSIS
==============================
🛡️  DB Verified: 2025-11-24
✅  GPU Driver Found: 535.129 (Supports CUDA 12.2)
✅  System CUDA (nvcc): 11.8

📦 Found torch: v2.2.1
   -> Bundled CUDA: 12.1
   ✅ Compatible with Driver.

🏭 COMPILATION HEALTH
❌ ASYMMETRY DETECTED: System (11.8) != Torch (12.1)
   -> pip install flash-attention will likely FAIL.
```

### 2. Get the Safe Install Command
Stop guessing which `cuXX` wheel works on your machine.

```console
$ doctor install torch

Detected Driver: 535.129 (Supports up to CUDA 12.2)

⬇️ Run this command to install the SAFE version:
---------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
---------------------------------------------------
```

### 3. Scan for Deprecated Code
Automatically detects outdated imports and suggests fixes.

```console
$ doctor scan

🦜 CODE MIGRATION CHECK
❌ Deprecated in src/main.py:4
   Found: 'langchain.chat_models'
   Moved to: 'langchain_community.chat_models'
```

## 🧩 Architecture

Env-Doctor relies on a self-improving database:

1.  **Scraper**: Watches PyTorch & NVIDIA release notes.
2.  **Validator**: Physically tests new versions on cloud GPUs.
3.  **Hybrid Cache**: The CLI fetches the latest rules from GitHub but works offline using cached data.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.