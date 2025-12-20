# 🩺 Env-Doctor: The AI Environment Fixer

![PyPI - Version](https://img.shields.io/pypi/v/env-doctor)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/env-doctor)
![License](https://img.shields.io/github/license/mitulgarg/env-doctor)
![CI Status](https://img.shields.io/github/actions/workflow/status/mitulgarg/env-doctor/ci.yml)

**Stop guessing which PyTorch version works with your NVIDIA driver.**

Env-Doctor is a CLI tool that bridges the gap between your hardware (NVIDIA drivers) and your software (Python AI libraries). It scans your system, detects your GPU driver version, and tells you **exactly** which pre-compiled binaries (wheels) will work — preventing random crashes and cryptic CUDA errors.

## 🚀 Features

*   **⚡ Automated Diagnosis**: Instantly checks compatibility between your GPU Driver (Kernel), System CUDA (Compiler), and Python Libs (Torch, TensorFlow, JAX).
*   **🛠️ Compilation Guard**: Warns if your system `nvcc` doesn't match `torch` bundled CUDA, preventing build failures for libs like `flash-attention`.
*   **🦜 Migration Helper**: Scans your code for deprecated imports (e.g., old LangChain or Pydantic schemas) and suggests fixes.
*   **🛡️ Verified Compatibility**: Uses a hybrid Verified Database (scraped & tested daily) to recommend safe installation commands.

## 📦 Installation

To install Env-Doctor, simply run:

```bash
pip install env-doctor
```

## 🛠️ Usage

### 1️⃣ Diagnose Your Environment
Check your current system health, driver info, and installed library conflicts.

```bash
env-doctor check
```

**What it checks:**
*   **GPU Driver**: Is it too old for your installed PyTorch?
*   **System CUDA**: Is it missing or mismatched?
*   **Library Conflicts**: Do you have a "Frankenstein" environment (e.g., Torch 2.1 with CUDA 12.1 vs Driver supporting only 11.8)?

### 2️⃣ Get the Safe Install Command
Don't guess which index-url to use. Let the doctor prescribe it.

```bash
env-doctor install torch
```

*Output Example:*
```bash
⬇️ Run this command to install the SAFE version:
---------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
---------------------------------------------------
```

### 3️⃣ Scan for Code Issues
Scan your project for deprecated imports (like old LangChain definitions).

```bash
env-doctor scan
```

## 🧩 Architecture

*   **The Brain**: `compatibility.json` maps drivers to max supported CUDA versions and verified wheel URLs.
*   **The Hands**: `checks.py` inspects the local machine using NVML and Python introspection.
*   **The Updater**: `db.py` fetches the latest rules from GitHub so you don't need to update the package daily.

## 🤝 Contributing

We love contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests and our development setup.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.