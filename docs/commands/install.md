# install

Get the safe install command for GPU libraries that matches your driver.

## Usage

```bash
env-doctor install <library>
```

## Supported Libraries

- `torch` / `pytorch` - PyTorch with correct CUDA wheels
- `tensorflow` - TensorFlow with GPU support
- `jax` - JAX with CUDA support
- `vllm`, `sglang` - inference engines, resolved to a build that runs on your driver (optionally pin a version: `vllm@0.19.1`, `sglang==0.5.19`)

## How It Works

1. Detects your NVIDIA driver version
2. Determines the maximum CUDA version your driver supports
3. Looks up the correct wheel URL for that CUDA version
4. Outputs the exact `pip install` command

## Example

```bash
env-doctor install torch
```

**Output:**

```
🔍 Detecting your GPU environment...

🎮 GPU Driver: 535.146.02
   └─ Max CUDA: 12.2

⬇️ Run this command to install the SAFE version:
---------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
---------------------------------------------------

💡 This installs PyTorch built for CUDA 12.1, which is compatible
   with your driver (supports up to CUDA 12.2).
```

## Why This Matters

The default `pip install torch` gives you the latest wheel, which might be built for CUDA 12.4. If your driver only supports CUDA 11.8, you'll get cryptic errors like:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Env-Doctor prevents this by prescribing the correct version for your hardware.

## Common Scenarios

### Older Driver

```bash
$ env-doctor install torch

🎮 GPU Driver: 470.82.01
   └─ Max CUDA: 11.4

⬇️ Run this command to install the SAFE version:
---------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113
---------------------------------------------------
```

### No GPU Detected

```bash
$ env-doctor install torch

⚠️  No NVIDIA GPU detected

⬇️ Installing CPU-only version:
---------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
---------------------------------------------------
```

## Inference Engines (vLLM / SGLang)

`pip install vllm` installs the default wheel, which since vLLM 0.20 targets CUDA 13 — it won't run on a CUDA 12.x driver. `env-doctor install vllm` picks the newest build that runs on *your* driver instead, preferring an alternate CUDA build of the same version over a version change, and a version change over a (risky) driver upgrade:

```bash
env-doctor install vllm            # newest vLLM, correct CUDA build
env-doctor install sglang@0.5.19   # a specific version
env-doctor install vllm --execute  # run the recommended commands
```

Commands use `uv` (needed for `--torch-backend`); if `uv` isn't on PATH, `pip install uv` is prepended. `--execute` never runs a driver-upgrade option. See [check](check.md#inference-engines-vllm-sglang) for how resolution works.

## See Also

- [check](check.md) - Full environment diagnosis
- [model](model.md) - Check model VRAM requirements
