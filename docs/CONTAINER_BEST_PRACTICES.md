# Container Best Practices for GPU Workloads

This guide explains best practices for running GPU workloads in Docker containers, and why the `env-doctor` validators flag certain patterns.

## Table of Contents
- [Base Images](#base-images)
- [PyTorch Installation](#pytorch-installation)
- [TensorFlow Installation](#tensorflow-installation)
- [Driver Management](#driver-management)
- [CUDA Toolkit](#cuda-toolkit)
- [Docker Compose Configuration](#docker-compose-configuration)
- [Host System Requirements](#host-system-requirements)

---

## Base Images

### ✅ Use GPU-Enabled Base Images

**Always use a CUDA-enabled base image for GPU workloads.**

```dockerfile
# ✅ GOOD - NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ✅ GOOD - PyTorch official image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ✅ GOOD - TensorFlow GPU image
FROM tensorflow/tensorflow:latest-gpu
```

**Why?** These images include the CUDA runtime libraries needed to communicate with the GPU driver.

### ❌ Don't Use CPU-Only Images

```dockerfile
# ❌ BAD - CPU-only image
FROM python:3.10

# ❌ BAD - Generic Ubuntu
FROM ubuntu:22.04

# ❌ BAD - Alpine (no CUDA support)
FROM alpine:latest
```

**Why?** These images lack CUDA runtime libraries. Your GPU code will fail at runtime with errors like:
- `CUDA not available`
- `No CUDA-capable device detected`
- `Could not load dynamic library 'libcudart.so'`

### Runtime vs Devel Images

Choose the right image variant:

- **Runtime** (`-runtime`): For running pre-trained models. Smaller size (~2-3GB).
- **Devel** (`-devel`): For compiling CUDA code (flash-attention, xformers). Larger size (~6-8GB).

```dockerfile
# For inference only
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# For training with custom CUDA extensions
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
```

---

## PyTorch Installation

### ✅ Always Use --index-url

**Never install PyTorch without specifying the CUDA version.**

```dockerfile
# ✅ GOOD - Explicit CUDA version
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ✅ GOOD - CUDA 11.8
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ Don't Install Without --index-url

```dockerfile
# ❌ BAD - May download wrong CUDA version
RUN pip install torch torchvision

# ❌ BAD - Defaults to latest, may mismatch driver
RUN pip install torch
```

**Why?** Without `--index-url`, pip defaults to the latest CUDA version (often 12.4+), which may not match:
- Your base image's CUDA version
- Your host driver's supported CUDA version

This causes runtime errors like:
- `CUDA driver version is insufficient`
- `no kernel image is available for execution`

### Match Base Image CUDA Version

If your base image is `nvidia/cuda:11.8`, use `cu118`:

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Version Mapping

| Base Image CUDA | --index-url suffix |
|-----------------|-------------------|
| 11.7.x          | cu117             |
| 11.8.x          | cu118             |
| 12.0.x          | cu120             |
| 12.1.x          | cu121             |
| 12.2.x          | cu122             |

---

## TensorFlow Installation

### ✅ Use tensorflow[and-cuda]

```dockerfile
# ✅ GOOD - Explicit GPU support
RUN pip install tensorflow[and-cuda]

# ✅ ALSO GOOD - Plain tensorflow with CUDA available
RUN pip install tensorflow
```

### ⚠️ Avoid tensorflow-gpu

```dockerfile
# ⚠️ DEPRECATED - Old package name
RUN pip install tensorflow-gpu
```

**Why?** `tensorflow-gpu` is deprecated. Modern TensorFlow (2.0+) automatically uses GPU when available. Use `tensorflow[and-cuda]` to ensure CUDA dependencies are installed.

---

## Driver Management

### ❌ NEVER Install NVIDIA Drivers in Containers

```dockerfile
# ❌ WRONG - Don't install drivers
RUN apt-get install -y nvidia-driver-535

# ❌ WRONG - Don't install any nvidia-driver-*
RUN apt-get install -y nvidia-driver-*
```

**Why?**
1. **Drivers must be on the host**, not in containers
2. Container shares host's kernel and GPU driver
3. Installing drivers in container:
   - Wastes space (~500MB)
   - Can cause conflicts
   - Won't work (container can't load kernel modules)

### ✅ Correct Driver Setup

- **Host**: Install NVIDIA driver on host machine
- **Container**: Use CUDA-enabled base image (has runtime libraries)
- **Docker**: Install `nvidia-container-toolkit` on host

```bash
# On host machine (Ubuntu/Debian)
sudo apt-get install -y nvidia-driver-535
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## CUDA Toolkit

### ⚠️ Toolkit Usually Not Needed

Most containers don't need the full CUDA Toolkit.

```dockerfile
# ⚠️ USUALLY UNNECESSARY - Adds 2-5GB
RUN apt-get install -y cuda-toolkit-12-1
```

**Why?**
- Runtime-only containers have everything needed to run GPU code
- Toolkit is only for **compiling** CUDA code
- Adds significant image bloat

### ✅ Only Install If Compiling

Install toolkit only when building from source:

```dockerfile
# ✅ GOOD - Needed for flash-attention
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-1 \
    build-essential

RUN pip install flash-attn --no-build-isolation
```

**When you need toolkit:**
- Building `flash-attention`
- Building `xformers`
- Compiling custom CUDA kernels
- Building `auto-gptq` from source

**When you DON'T need it:**
- Installing PyTorch/TensorFlow wheels (pre-compiled)
- Running inference
- Running training with standard ops

---

## Docker Compose Configuration

### ✅ Modern GPU Configuration (Compose v2.3+)

```yaml
version: '3.8'

services:
  gpu-app:
    image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # or 1, 2, etc.
              capabilities: [gpu]
    command: python train.py
```

**Required fields:**
- `driver: nvidia` - Specifies NVIDIA GPU
- `count: all` or specific number - How many GPUs
- `capabilities: [gpu]` - Enable GPU capability

### ❌ Deprecated Syntax

```yaml
# ❌ OLD - Deprecated in Compose v2.3+
services:
  old-app:
    image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
    runtime: nvidia
```

**Why deprecated?**
- Less flexible (can't control GPU allocation)
- Removed in Compose v3+
- Doesn't support advanced features (device selection, memory limits)

### Multi-GPU Services

Assign specific GPUs to different services:

```yaml
version: '3.8'

services:
  training:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']  # First two GPUs
              capabilities: [gpu]

  inference:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']  # Third GPU
              capabilities: [gpu]
```

---

## Host System Requirements

### Prerequisites

Before running GPU containers, ensure your host has:

1. **NVIDIA Driver**
   ```bash
   nvidia-smi  # Should show driver version and GPUs
   ```

2. **nvidia-container-toolkit**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y nvidia-container-toolkit

   # RHEL/CentOS
   sudo yum install -y nvidia-container-toolkit
   ```

3. **Docker Daemon Configuration**

   Create/edit `/etc/docker/daemon.json`:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

   Restart Docker:
   ```bash
   sudo systemctl restart docker
   ```

### Verify Setup

Test GPU access in a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

Should display GPU information.

---

## Common Issues and Solutions

### Issue: "CUDA not available" in container

**Causes:**
- Using CPU-only base image
- nvidia-container-toolkit not installed
- Missing `--gpus all` flag (when running `docker run`)
- Missing GPU device config (in docker-compose)

**Solutions:**
1. Use CUDA-enabled base image
2. Install nvidia-container-toolkit on host
3. Add GPU configuration to docker-compose.yml

### Issue: "CUDA driver version insufficient"

**Causes:**
- PyTorch/TensorFlow built for newer CUDA than driver supports
- Missing `--index-url` in pip install

**Solution:**
```bash
# Check driver's max CUDA version
nvidia-smi | grep "CUDA Version"

# Install matching PyTorch
# If driver supports CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Image too large

**Causes:**
- Using `-devel` images when `-runtime` is sufficient
- Installing CUDA toolkit unnecessarily

**Solutions:**
1. Use `-runtime` images for inference
2. Only install toolkit when compiling CUDA code
3. Use multi-stage builds to copy only needed artifacts

---

## Quick Validation

Before building your container, run:

```bash
# Validate Dockerfile
env-doctor dockerfile

# Validate docker-compose.yml
env-doctor docker-compose
```

Fix all errors before building to avoid runtime issues.

---

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose GPU Access](https://docs.docker.com/compose/gpu-support/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [TensorFlow Docker Images](https://hub.docker.com/r/tensorflow/tensorflow)
