# Getting Started

This guide will help you install Env-Doctor and run your first environment check.

## Prerequisites

- Python 3.7 or higher
- NVIDIA GPU (optional, but recommended)
- NVIDIA driver installed (for GPU features)

## Installation

### From PyPI (Recommended)

```bash
pip install env-doctor
```

### From Source

```bash
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor
pip install -e .
```

## Your First Check

Run the environment check to see your current GPU/CUDA status:

```bash
env-doctor check
```

This command checks:

- **Environment**: Native Linux vs WSL1 vs WSL2, with GPU forwarding validation
- **GPU Driver**: Version and maximum supported CUDA version
- **System CUDA**: Installation status and version
- **Library Conflicts**: Detects mismatched versions (e.g., PyTorch built for CUDA 12.4 on a driver supporting only CUDA 11.8)
- **WSL2 GPU Setup**: Validates CUDA libraries and driver configuration

### Example Output

```
🩺 ENV-DOCTOR DIAGNOSIS
============================================================

🖥️  Environment: Native Linux

🎮 GPU Driver
   ✅ NVIDIA Driver: 535.146.02
   └─ Max CUDA: 12.2

🔧 CUDA Toolkit
   ✅ System CUDA: 12.1.1
   └─ Path: /usr/local/cuda-12.1

📦 Python Libraries
   ✅ torch 2.1.0+cu121
   ✅ tensorflow 2.15.0

✅ All checks passed! Your environment is healthy.
```

## Common Next Steps

### Get a Safe Install Command

If you need to install or reinstall PyTorch:

```bash
env-doctor install torch
```

This outputs the exact `pip install` command with the correct `--index-url` for your driver.

### Check Model Compatibility

Before downloading a large model:

```bash
env-doctor model llama-3-8b
```

### Validate Your Dockerfile

Before building a GPU container:

```bash
env-doctor dockerfile path/to/Dockerfile
```

## Quick Command Reference

| Command | Purpose |
|---------|---------|
| `env-doctor check` | Full environment diagnosis |
| `env-doctor install <lib>` | Get safe install command |
| `env-doctor model <name>` | Check model VRAM requirements |
| `env-doctor cuda-info` | Detailed CUDA toolkit analysis |
| `env-doctor cudnn-info` | cuDNN library analysis |
| `env-doctor dockerfile` | Validate Dockerfile |
| `env-doctor docker-compose` | Validate docker-compose.yml |
| `env-doctor scan` | Scan for deprecated imports |
| `env-doctor debug` | Verbose detector output |

## Next Steps

- [Command Reference](commands/check.md) - Detailed documentation for each command
- [WSL2 Guide](guides/wsl2.md) - GPU setup for Windows Subsystem for Linux
- [CI/CD Integration](guides/ci-cd.md) - Use Env-Doctor in your pipelines