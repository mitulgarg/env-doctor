<p align="center">
  <img src="https://raw.githubusercontent.com/mitulgarg/env-doctor/main/docs/assets/logo.svg" alt="Env-Doctor Logo" width="80" height="80">
</p>

<h1 align="center">Env-Doctor</h1>


<p align="center">
  <strong>The missing link between your GPU and Python AI libraries</strong>
</p>

<p align="center">
  <a href="https://mitulgarg.github.io/env-doctor/">
    <img src="https://img.shields.io/badge/docs-github.io-blueviolet?style=flat-square" alt="Documentation">
  </a>
  <a href="https://pypi.org/project/env-doctor/">
    <img src="https://img.shields.io/pypi/v/env-doctor?style=flat-square&color=blue&label=PyPI" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/env-doctor/">
    <img src="https://img.shields.io/pypi/dm/env-doctor?style=flat-square&color=success&label=Downloads" alt="Downloads">
  </a>
  <img src="https://img.shields.io/badge/python-3.7+-blue?style=flat-square" alt="Python">
  <a href="https://github.com/mitulgarg/env-doctor/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/mitulgarg/env-doctor?style=flat-square&color=green" alt="License">
  </a>
  <a href="https://github.com/mitulgarg/env-doctor/stargazers">
    <img src="https://img.shields.io/github/stars/mitulgarg/env-doctor?style=flat-square&color=yellow" alt="GitHub Stars">
  </a>
</p>

---

> **"Why does my PyTorch crash with CUDA errors when I just installed it?"**
>
> Because your driver supports CUDA 11.8, but `pip install torch` gave you CUDA 12.4 wheels.

**Env-Doctor diagnoses and fixes the #1 frustration in GPU computing:** mismatched CUDA versions between your NVIDIA driver, system toolkit, cuDNN, and Python libraries.

It takes **5 seconds** to find out if your environment is broken - and exactly how to fix it.


## Doctor "Check" (Diagnosis)

![Env-Doctor Demo](https://raw.githubusercontent.com/mitulgarg/env-doctor/main/docs/assets/envdoctordemo.gif)


## Features

| Feature | What It Does |
|---------|--------------|
| **One-Command Diagnosis** | Check compatibility: GPU Driver → CUDA Toolkit → cuDNN → PyTorch/TensorFlow/JAX |
| **Safe Install Commands** | Get the exact `pip install` command that works with YOUR driver |
| **Extension Library Support** | Install compilation packages (flash-attn, SageAttention, auto-gptq, apex, xformers) with CUDA version matching |
| **AI Model Compatibility** | Check if LLMs, Diffusion, or Audio models fit on your GPU before downloading |
| **WSL2 GPU Support** | Validate GPU forwarding, detect driver conflicts within WSL2 env for Windows users |
| **Deep CUDA Analysis** | Find multiple installations, PATH issues, environment misconfigurations |
| **Container Validation** | Catch GPU config errors in Dockerfiles before you build |
| **CI/CD Ready** | JSON output and proper exit codes for automation |

## Installation

```bash
pip install env-doctor
```

Or from source:

```bash
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor
pip install -e .
```

## Usage

### Diagnose Your Environment

```bash
env-doctor check
```

**Example output:**
```
🩺 ENV-DOCTOR DIAGNOSIS
============================================================

🖥️  Environment: Native Linux

🎮 GPU Driver
   ✅ NVIDIA Driver: 535.146.02
   └─ Max CUDA: 12.2

🔧 CUDA Toolkit
   ✅ System CUDA: 12.1.1

📦 Python Libraries
   ✅ torch 2.1.0+cu121

✅ All checks passed!
```

### Get Safe Install Command

```bash
env-doctor install torch
```

```
⬇️ Run this command to install the SAFE version:
---------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
---------------------------------------------------
```

### Install Compilation Packages (Extension Libraries)

For extension libraries like **flash-attn**, **SageAttention**, **auto-gptq**, **apex**, and **xformers** that require compilation from source, `env-doctor` provides special guidance to handle CUDA version mismatches:

```bash
env-doctor install flash-attn
```

**Example output (with CUDA mismatch):**
```
🩺  PRESCRIPTION FOR: flash-attn

⚠️   CUDA VERSION MISMATCH DETECTED
     System nvcc: 12.1.1
     PyTorch CUDA: 12.4.1

🔧  flash-attn requires EXACT CUDA version match for compilation.
    You have TWO options to fix this:

============================================================
📦  OPTION 1: Install PyTorch matching your nvcc (12.1)
============================================================

Trade-offs:
  ✅ No system changes needed
  ✅ Faster to implement
  ❌ Older PyTorch version (may lack new features)

Commands:
  # Uninstall current PyTorch
  pip uninstall torch torchvision torchaudio -y

  # Install PyTorch for CUDA 12.1
  pip install torch --index-url https://download.pytorch.org/whl/cu121

  # Install flash-attn
  pip install flash-attn --no-build-isolation

============================================================
⚙️   OPTION 2: Upgrade nvcc to match PyTorch (12.4)
============================================================

Trade-offs:
  ✅ Keep latest PyTorch
  ✅ Better long-term solution
  ❌ Requires system-level changes
  ❌ Verify driver supports CUDA 12.4

Steps:
  1. Check driver compatibility:
     env-doctor check

  2. Download CUDA Toolkit 12.4:
     https://developer.nvidia.com/cuda-12-4-0-download-archive

  3. Install CUDA Toolkit (follow NVIDIA's platform-specific guide)

  4. Verify installation:
     nvcc --version

  5. Install flash-attn:
     pip install flash-attn --no-build-isolation

============================================================
```

### Check Model Compatibility

```bash
env-doctor model llama-3-8b
```

```
🤖  Checking: LLAMA-3-8B (8.0B params)

🖥️   Your Hardware: RTX 3090 (24GB)

💾  VRAM Requirements:
  ✅  FP16: 19.2GB - fits with 4.8GB free
  ✅  INT4:  4.8GB - fits with 19.2GB free

✅  This model WILL FIT on your GPU!
```

List all models: `env-doctor model --list`

Automatic HuggingFace Support (New ✨)
If a model isn't found locally, env-doctor automatically checks the HuggingFace Hub, fetches its parameter metadata, and caches it locally for future runs — no manual setup required.

```bash
# Fetches from HuggingFace on first run, cached afterward
env-doctor model bert-base-uncased
env-doctor model sentence-transformers/all-MiniLM-L6-v2
```

**Output:**

```
🤖  Checking: BERT-BASE-UNCASED
    (Fetched from HuggingFace API - cached for future use)
    Parameters: 0.11B
    HuggingFace: bert-base-uncased

🖥️   Your Hardware:
    RTX 3090 (24GB VRAM)

💾  VRAM Requirements & Compatibility
  ✅  FP16:  264 MB - Fits easily!

💡  Recommendations:
1. Use fp16 for best quality on your GPU
```



### Validate Dockerfiles

```bash
env-doctor dockerfile
```

```
🐳  DOCKERFILE VALIDATION

❌  Line 1: CPU-only base image: python:3.10
    Fix: FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

❌  Line 8: PyTorch missing --index-url
    Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### More Commands

| Command | Purpose |
|---------|---------|
| `env-doctor check` | Full environment diagnosis |
| `env-doctor install <lib>` | Safe install command for PyTorch/TensorFlow/JAX, extension libraries (flash-attn, auto-gptq, apex, xformers, SageAttention, etc.) |
| `env-doctor model <name>` | Check model VRAM requirements |
| `env-doctor cuda-info` | Detailed CUDA toolkit analysis |
| `env-doctor cudnn-info` | cuDNN library analysis |
| `env-doctor dockerfile` | Validate Dockerfile |
| `env-doctor docker-compose` | Validate docker-compose.yml |
| `env-doctor scan` | Scan for deprecated imports |
| `env-doctor debug` | Verbose detector output |

### CI/CD Integration

```bash
# JSON output for scripting
env-doctor check --json

# CI mode with exit codes (0=pass, 1=warn, 2=error)
env-doctor check --ci
```

**GitHub Actions example:**
```yaml
- run: pip install env-doctor
- run: env-doctor check --ci
```

## Documentation

**Full documentation:** https://mitulgarg.github.io/env-doctor/

- [Getting Started](docs/getting-started.md)
- [Command Reference](docs/commands/check.md)
- [WSL2 GPU Guide](docs/guides/wsl2.md)
- [CI/CD Integration](docs/guides/ci-cd.md)
- [Architecture](docs/architecture.md)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE)
