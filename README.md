# 🩺 Env-Doctor: The AI Environment Fixer

![PyPI - Version](https://img.shields.io/pypi/v/env-doctor)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/env-doctor)
![License](https://img.shields.io/github/license/mitulgarg/env-doctor)
![CI Status](https://img.shields.io/github/actions/workflow/status/mitulgarg/env-doctor/ci.yml)

**Stop guessing which PyTorch version works with your NVIDIA driver.**

Env-Doctor is a CLI tool that bridges the gap between your hardware (NVIDIA drivers) and your software (Python AI libraries). It scans your system, detects your GPU driver version, and tells you **exactly** which pre-compiled binaries (wheels) will work — preventing random crashes and cryptic CUDA errors.

## 🚀 Features

*   **⚡ Automated Diagnosis**: Instantly checks compatibility between your GPU Driver (Kernel), System CUDA (Compiler), and Python Libs (Torch, TensorFlow, JAX).
*   **🐧 WSL2 GPU Support**: Detects WSL2 environments and validates GPU forwarding setup, including CUDA library presence and internal driver conflicts.
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
*   **Environment**: Native Linux vs WSL1 vs WSL2, with GPU forwarding validation for WSL2
*   **GPU Driver**: Is it too old for your installed PyTorch?
*   **System CUDA**: Is it missing or mismatched?
*   **Library Conflicts**: Do you have a "Frankenstein" environment (e.g., Torch 2.1 with CUDA 12.1 vs Driver supporting only 11.8)?
*   **WSL2 GPU Setup**: Validates CUDA libraries, checks for driver conflicts, tests nvidia-smi functionality

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

### 4️⃣ Debug Mode (Troubleshooting)
Get detailed information from all detectors for troubleshooting and development.

```bash
env-doctor debug
```

**What debug mode shows:**
- **All Detector Results**: Raw output from every registered detector
- **Detection Metadata**: Internal detection methods, paths, and detailed status
- **Registry Information**: List of all available detectors
- **Error Details**: Full exception traces and diagnostic information

*Example Output:*
```bash
🔍 DEBUG MODE - Detailed Detector Information
============================================================
Registered Detectors: cuda_toolkit, nvidia_driver, python_library, wsl2

--- WSL2 ---
Status: Status.SUCCESS
Component: wsl2
Version: wsl2
Metadata: {'environment': 'WSL2', 'gpu_forwarding': 'enabled'}

--- NVIDIA DRIVER ---
Status: Status.SUCCESS
Component: nvidia_driver
Version: 535.146.02
Path: /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
Metadata: {'max_cuda_version': '12.2', 'detection_method': 'nvml'}
```

**Use debug mode when:**
- Environment diagnosis shows unexpected results
- You want to understand what each detector found
- Contributing to the project or reporting issues
- Validating detector behavior in different environments

### 🐧 WSL2 GPU Support

Env-Doctor provides comprehensive WSL2 environment detection and GPU forwarding validation:

**Environment Detection:**
- **Native Linux**: Standard Linux environment detection
- **WSL1**: Detects WSL1 and warns that CUDA is not supported at all
- **WSL2**: Full GPU forwarding validation and troubleshooting

**WSL2 GPU Validation:**
- ✅ **Driver Conflicts**: Detects problematic internal NVIDIA drivers that break GPU forwarding
- ✅ **CUDA Libraries**: Validates presence of `/usr/lib/wsl/lib/libcuda.so`
- ✅ **nvidia-smi**: Tests functionality and provides specific error guidance
- ✅ **Recommendations**: Provides actionable steps to fix GPU forwarding issues

**Common WSL2 Issues Detected:**
```bash
❌ NVIDIA driver installed inside WSL. This breaks GPU forwarding.
   → Run: sudo apt remove --purge nvidia-*

❌ Missing /usr/lib/wsl/lib/libcuda.so
   → Reinstall NVIDIA driver on Windows host

❌ nvidia-smi command failed
   → Install NVIDIA driver on Windows (version 470.76 or newer)
```

## 📋 Quick Command Reference

```bash
env-doctor check              # Diagnose your environment
env-doctor install torch      # Get safe install command for PyTorch  
env-doctor scan               # Scan project for AI library imports
env-doctor debug              # Show detailed detector information
```

## 🧩 Architecture

*   **The Brain**: `compatibility.json` maps drivers to max supported CUDA versions and verified wheel URLs.
*   **The Detectors**: Modular detection system with specialized detectors for:
    - `WSL2Detector`: Environment detection and GPU forwarding validation
    - `NvidiaDriverDetector`: GPU driver version and capability detection
    - `CudaToolkitDetector`: System CUDA installation detection
    - `PythonLibraryDetector`: Python AI library version and CUDA compatibility
*   **The Registry**: `DetectorRegistry` provides a plugin system for easy detector discovery and execution.
*   **The CLI**: `cli.py` orchestrates all detectors and presents unified diagnostics.
*   **The Updater**: `db.py` fetches the latest rules from GitHub so you don't need to update the package daily.

## 🤝 Contributing

We love contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests and our development setup.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.