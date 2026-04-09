# Getting Started

This guide will help you install Env-Doctor and run your first environment check.

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU (optional, but recommended)
- NVIDIA driver installed (for GPU features)

## Installation

### Core CLI

```bash
pip install env-doctor
```

**Or with [uv](https://docs.astral.sh/uv/):**

```bash
uv tool install env-doctor   # isolated tool install
uvx env-doctor check         # run once without installing
```

### With Fleet Dashboard *(optional)*

If you want to run the web dashboard to monitor multiple machines, install the `dashboard` extra:

```bash
pip install env-doctor[dashboard]
```

This adds FastAPI, uvicorn, SQLAlchemy, and aiosqlite. The core CLI commands are identical either way — the extra only enables `env-doctor dashboard` and `env-doctor report`.

!!! note
    The dashboard is an observability tool for teams with multiple GPU machines. If you're working on a single machine, the core install is all you need.

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

### Get CUDA Toolkit Installation Instructions

If you need to install CUDA Toolkit on your system:

```bash
env-doctor cuda-install
```

This provides platform-specific, copy-paste installation commands for your system (Ubuntu, Debian, RHEL, Fedora, WSL2, Windows, Conda).

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
| `env-doctor cuda-install` | Get CUDA Toolkit installation guide |
| `env-doctor install <lib>` | Get safe install command |
| `env-doctor model <name>` | Check model VRAM requirements |
| `env-doctor cuda-info` | Detailed CUDA toolkit analysis |
| `env-doctor cudnn-info` | cuDNN library analysis |
| `env-doctor wsl` | Detailed WSL environment analysis |
| `env-doctor dockerfile` | Validate Dockerfile |
| `env-doctor docker-compose` | Validate docker-compose.yml |
| `env-doctor scan` | Scan for deprecated imports |
| `env-doctor debug` | Verbose detector output |
| `env-doctor dashboard` | Start fleet web UI *(requires `[dashboard]` extra)* |
| `env-doctor report install` | Set up periodic reporting to a dashboard *(requires `[dashboard]` extra)* |

## Next Steps

- [Command Reference](commands/check.md) - Detailed documentation for each command
- [WSL2 Guide](guides/wsl2.md) - GPU setup for Windows Subsystem for Linux
- [CI/CD Integration](guides/ci-cd.md) - Use Env-Doctor in your pipelines
- [Fleet Monitoring](guides/fleet-monitoring.md) - Monitor multiple GPU machines from a web UI