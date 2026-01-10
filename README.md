# 🩺 Env-Doctor

### **The missing link between your GPU and Python AI libraries**

![License](https://img.shields.io/github/license/mitulgarg/env-doctor)
![CI Status](https://img.shields.io/github/actions/workflow/status/mitulgarg/env-doctor/ci.yml)
![Python](https://img.shields.io/badge/python-3.7+-blue)

---

> **"Why does my PyTorch crash with CUDA errors when I just installed it?"**
>
> Because your driver supports CUDA 11.8, but `pip install torch` gave you CUDA 12.4 wheels.

---

**Env-Doctor diagnoses and fixes the #1 frustration in GPU computing:** mismatched CUDA versions between your NVIDIA driver, system toolkit, cuDNN, and Python libraries.

It takes **5 seconds** to find out if your environment is broken — and exactly how to fix it.

## 🚀 Features

| Feature | What It Does |
|---------|--------------|
| **⚡ One-Command Diagnosis** | Instantly check compatibility between GPU Driver → CUDA Toolkit → cuDNN → PyTorch/TensorFlow/JAX |
| **🔧 Deep CUDA Analysis** | `cuda-info` reveals multiple installations, PATH issues, environment misconfigurations |
| **🧠 cuDNN Detection** | `cudnn-info` finds cuDNN libraries, validates symlinks, checks version compatibility |
| **🤖 AI Model Compatibility** | Check if your GPU can run any model (LLMs, Diffusion, Audio) before downloading |
| **🐧 WSL2 GPU Support** | Detects WSL1/WSL2 environments, validates GPU forwarding, catches common driver conflicts |
| **🛠️ Compilation Guard** | Warns if system `nvcc` doesn't match PyTorch's CUDA — preventing flash-attention build failures |
| **💊 Safe Install Commands** | Prescribes the exact `pip install` command that works with YOUR driver |
| **🦜 Migration Helper** | Scans code for deprecated imports (LangChain, Pydantic) and suggests fixes |

## 📦 Installation

> ⏳ **Coming Soon**: `pip install env-doctor` will be available shortly!

For now, install from source:

```bash
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor
pip install -e .
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

### 4️⃣ Deep CUDA Toolkit Analysis
Get comprehensive details about your CUDA installations, environment variables, and configuration issues.

```bash
env-doctor cuda-info
```

**What it shows:**
*   Multiple CUDA installations and their paths
*   `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH` configuration
*   Runtime library (libcudart) status
*   Driver compatibility with installed toolkit

### 5️⃣ cuDNN Library Analysis
Detect cuDNN installations and validate they're properly configured.

```bash
env-doctor cudnn-info
```

**What it shows:**
*   cuDNN version and library locations
*   Multiple installation detection
*   Symlink validation (Linux)
*   PATH configuration (Windows)
*   CUDA compatibility status

### 6️⃣ Check AI Model Compatibility
Before downloading a 40GB+ model, find out if it will run on your GPU!

```bash
env-doctor model llama-3-8b
```

**What it checks:**
*   **Model Parameters**: LLMs, Diffusion models, Audio models, and more
*   **VRAM Requirements**: Calculates VRAM needed for each precision (fp32, fp16, int8, int4)
*   **GPU Availability**: Detects your GPU and available VRAM
*   **Compatibility Analysis**: Shows which precisions fit on your hardware
*   **Smart Recommendations**: Suggests smaller variants or multi-GPU setup if needed

**List all available models:**
```bash
env-doctor model --list
```

**Check specific precision:**
```bash
env-doctor model stable-diffusion-xl --precision int4
```

**Example Output:**
```
🤖  Checking: LLAMA-3-8B
    Parameters: 8.0B
    HuggingFace: meta-llama/Meta-Llama-3-8B

🖥️   Your Hardware:
    RTX 3090 (24GB VRAM)

============================================================
💾  VRAM Requirements & Compatibility
============================================================

  ✅  FP16: 19.2GB (measured) - 4.8GB free
  ✅  INT4:  4.8GB (estimated) - 19.2GB free

============================================================
✅  This model WILL FIT on your GPU!
============================================================

💡  Recommendations:
1. Use fp16 for best quality on your GPU
```

**Supported Models:**
- **LLMs**: Llama-3, Mistral, Mixtral, Qwen (8B-405B parameters)
- **Diffusion**: Stable Diffusion 1.5/XL, Flux, Stable Diffusion 3
- **Audio**: Whisper (tiny to large-v3)
- **Language**: BERT, T5 (for embeddings and text encoding)

**Key Features:**
- Measured VRAM for popular models (most accurate)
- Formula-based estimation for new models
- Multi-GPU support (total VRAM calculation)
- Alias support (e.g., "sdxl" → "stable-diffusion-xl")
- Family variants (e.g., suggest llama-3-8b when 70b won't fit)

### 7️⃣ Debug Mode (Troubleshooting)
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
env-doctor cuda-info          # Detailed CUDA toolkit analysis
env-doctor cudnn-info         # Detailed cuDNN library analysis
env-doctor model llama-3-8b   # Check if model fits on your GPU
env-doctor model --list       # List all available models
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
    - `CudnnDetector`: cuDNN library detection and configuration validation
    - `PythonLibraryDetector`: Python AI library version and CUDA compatibility
*   **The Registry**: `DetectorRegistry` provides a plugin system for easy detector discovery and execution.
*   **The CLI**: `cli.py` orchestrates all detectors and presents unified diagnostics.
*   **The Updater**: `db.py` fetches the latest rules from GitHub so you don't need to update the package daily.

## 🤝 Contributing

We love contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests and our development setup.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.