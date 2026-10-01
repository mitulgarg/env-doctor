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

### GPU Compute Capability

Checks whether the installed PyTorch wheel includes compiled kernels for your GPU's SM architecture. This catches a silent failure mode common with new GPU generations: everything looks healthy (`nvidia-smi`, `nvcc`, driver all pass) but CUDA may not work correctly because the stable PyTorch wheel doesn't include kernels for the new architecture.

env-doctor probes `torch.cuda.is_available()` at runtime and distinguishes two failure modes:

- **Hard failure** — `is_available()` returns `False`. The GPU cannot be used at all.
- **Soft failure** — `is_available()` returns `True` via NVIDIA's driver-level PTX JIT, but complex CUDA ops may silently degrade or fail.

Other behaviours:
- Reads GPU compute capability from the driver (e.g. `12.0` for Blackwell RTX 5070)
- Reads the compiled SM list from `torch.cuda.get_arch_list()`
- Handles PTX forward compatibility — `compute_90` in the arch list covers newer SMs via JIT compilation
- On mismatch, prints the exact nightly install command to fix it

### Library Conflicts

Detects "Frankenstein" environments where:

- PyTorch is built for CUDA 12.4 but driver only supports 11.8
- Multiple libraries compiled for different CUDA versions
- System toolkit doesn't match library requirements

### Inference Engines (vLLM / SGLang)

Installed vLLM / SGLang are checked automatically. To check an engine *before* installing it, pass `--engine` (repeatable, optionally with a version):

```bash
env-doctor check --engine vllm
env-doctor check --engine vllm@0.20.0 --engine sglang
```

env-doctor resolves the whole chain — engine version → pinned torch → CUDA build of the wheels → kernel libs (FlashInfer, sgl-kernel) — against the CUDA version your **driver** supports. pip wheels bundle their own CUDA runtime, so the driver is the binding constraint; the system toolkit (nvcc) only matters for JIT-compiled kernels and is reported as a soft warning.

Fix options are ranked best-first:

1. **Same engine version, different CUDA build** (e.g. vLLM's `cu129` wheel index instead of the default CUDA 13 wheel)
2. **Newest engine version** with a build that runs on your driver (e.g. SGLang 0.5.19, the last release with a CUDA 12 lane)
3. **Driver upgrade** — last resort, flagged ⚠ RISKY because it can break other deployments on a shared box

A build for a newer CUDA *minor* version (e.g. 12.9 on a 12.6 driver) is allowed via NVIDIA's CUDA minor-version compatibility, with a warning; a newer *major* version (13.x on a 12.x driver) is not.

```
⚠️   🚀 vLLM 0.30.0  →  needs attention on driver CUDA 12.6
    Default wheel: CUDA 13.0, torch 2.13.0
    ❌ Default `pip install vllm==0.30.0` pulls a CUDA 13.0 build (torch 2.13.0), but the driver supports up to CUDA 12.6.
    → Fix options (best first):
      1. Install the CUDA 12.9 build of vLLM 0.30.0 — via CUDA minor-version compatibility
           uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129
      2. Upgrade the NVIDIA driver to ≥ 580 to use the default CUDA 13.0 build — on a shared box this can break other deployments  ⚠ RISKY
    📋 Copy-to-fix: uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129
```

In `--json` output the resolution appears under `checks.engines.<name>` (status, ranked options, `copy_to_fix`). The compatibility table lives in `data/inference_engines.json` and refreshes from GitHub like the main compatibility DB. See the [Inference Engines guide](../guides/inference-engines.md) for the full walkthrough.

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

### Compute Capability: Compatible

```
🎯  COMPUTE CAPABILITY CHECK
    GPU: NVIDIA GeForce GTX 1650 (Compute 7.5, Turing, sm_75)
    PyTorch compiled for: sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90
    ✅ COMPATIBLE: PyTorch 2.5.1+cu121 supports your GPU architecture.
```

### Compute Capability: Hard Mismatch (`is_available()` → `False`)

```
🎯  COMPUTE CAPABILITY CHECK
    GPU: NVIDIA GeForce RTX 5070 (Compute 12.0, Blackwell, sm_120)
    PyTorch compiled for: sm_50, sm_60, sm_70, sm_80, sm_90, compute_90
    ❌ ARCHITECTURE MISMATCH: Your GPU needs sm_120 but PyTorch 2.5.1 doesn't include it.

    This is likely why torch.cuda.is_available() returns False even though
    your driver and CUDA toolkit are working correctly.

    FIX: Install PyTorch nightly with sm_120 support:
       pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

### Compute Capability: Soft Mismatch (`is_available()` → `True` via PTX JIT)

```
🎯  COMPUTE CAPABILITY CHECK
    GPU: NVIDIA GeForce RTX 5070 (Compute 12.0, Blackwell, sm_120)
    PyTorch compiled for: sm_50, sm_60, sm_70, sm_80, sm_90, compute_90
    ⚠️  ARCHITECTURE MISMATCH (Soft): Your GPU needs sm_120 but PyTorch 2.5.1 doesn't include it.

    torch.cuda.is_available() returned True via NVIDIA's driver-level PTX JIT,
    but you may experience degraded performance or failures with complex CUDA ops.

    FIX: Install a newer PyTorch with native sm_120 support for full compatibility:
       pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
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
    },
    "compute_compatibility": {
      "gpu_name": "NVIDIA GeForce GTX 1650",
      "compute_capability": "7.5",
      "sm": "sm_75",
      "arch_name": "Turing",
      "arch_list": ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
      "status": "compatible",
      "cuda_available": true,
      "message": "PyTorch supports sm_75 (Turing)"
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
