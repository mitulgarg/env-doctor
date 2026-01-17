# check

Diagnose your environment for GPU/CUDA compatibility issues.

## Usage

```bash
env-doctor check
```

## What It Checks

### Environment Detection

- **Native Linux**: Standard Linux environment
- **WSL1**: Detects WSL1 and warns that CUDA is not supported
- **WSL2**: Full GPU forwarding validation

### GPU Driver

- Driver version detection via NVML
- Maximum supported CUDA version
- Driver health status

### System CUDA Toolkit

- Installation path and version
- Multiple installation detection
- PATH and environment configuration

### Python Libraries

- PyTorch, TensorFlow, JAX detection
- CUDA version each library was compiled for
- Compatibility with your driver

### Library Conflicts

Detects "Frankenstein" environments where:

- PyTorch is built for CUDA 12.4 but driver only supports 11.8
- Multiple libraries compiled for different CUDA versions
- System toolkit doesn't match library requirements

## Example Output

```
🩺 ENV-DOCTOR DIAGNOSIS
============================================================

🖥️  Environment: WSL2 (GPU forwarding enabled)

🎮 GPU Driver
   ✅ NVIDIA Driver: 535.146.02
   └─ Max CUDA: 12.2

🔧 CUDA Toolkit
   ✅ System CUDA: 12.1.1
   └─ Path: /usr/local/cuda-12.1

📦 Python Libraries
   ✅ torch 2.1.0+cu121
   └─ CUDA 12.1 ✓ (compatible with driver)

✅ All checks passed!
```

## Common Issues Detected

### Driver Too Old

```
❌ PyTorch requires CUDA 12.1 but driver only supports CUDA 11.8
   → Update your NVIDIA driver to 520.61.05 or newer
   → Or install PyTorch for CUDA 11.8:
     pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Missing CUDA Toolkit

```
⚠️  No system CUDA toolkit found
   → This is OK if you only use PyTorch/TensorFlow (they bundle CUDA)
   → Install CUDA toolkit if you need to compile extensions
```

### WSL2 GPU Issues

```
❌ NVIDIA driver installed inside WSL. This breaks GPU forwarding.
   → Run: sudo apt remove --purge nvidia-*
```

---

## Advanced Options

### JSON Output

For scripting or parsing results programmatically:

```bash
env-doctor check --json
```

```json
{
  "status": "success",
  "timestamp": "2026-01-15T10:30:00Z",
  "summary": {
    "driver": "found",
    "cuda": "found",
    "issues_count": 0
  },
  "checks": {
    "driver": {
      "component": "nvidia_driver",
      "status": "success",
      "detected": true,
      "version": "535.146.02",
      "metadata": {
        "max_cuda_version": "12.2"
      }
    },
    "cuda": {
      "component": "cuda_toolkit",
      "status": "success",
      "detected": true,
      "version": "12.1.1",
      "path": "/usr/local/cuda-12.1"
    },
    "libraries": {
      "torch": {
        "version": "2.1.0+cu121",
        "cuda_version": "12.1",
        "compatible": true
      }
    }
  }
}
```

### CI/CD Mode

For continuous integration pipelines:

```bash
env-doctor check --ci
```

This implies `--json` and sets proper exit codes:

| Code | Meaning |
|------|---------|
| `0` | All checks passed |
| `1` | Warnings or non-critical issues |
| `2` | Critical errors detected |

See [CI/CD Integration Guide](../guides/ci-cd.md) for full pipeline examples.

## See Also

- [cuda-info](cuda-info.md) - Detailed CUDA toolkit analysis
- [cudnn-info](cudnn-info.md) - cuDNN library analysis
- [debug](debug.md) - Verbose detector output
