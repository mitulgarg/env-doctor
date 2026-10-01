---
description: Install vLLM and SGLang without breaking CUDA. env-doctor resolves the engine → torch → CUDA build → kernel-lib chain against your NVIDIA driver and gives you the install command that actually runs.
---

# Inference Engines: vLLM & SGLang

`env-doctor` resolves **which build of vLLM or SGLang will actually run on your NVIDIA driver**, and gives you the exact install command for it. If the default `pip install` would break, it tells you why before you install anything.

```bash
env-doctor install vllm             # driver-compatible install command
env-doctor check --engine vllm      # diagnose before installing
env-doctor check                    # installed engines are checked automatically
```

*Available since v0.3.5.*

## The problem

On a box whose driver supports up to CUDA 12.x:

```bash
pip install vllm      # vLLM ≥ 0.20 → default wheel is built for CUDA 13.0
```

The install succeeds, but vLLM then fails at runtime: a CUDA 13 wheel can't run on a CUDA 12 driver. Each component looks fine on its own. The breakage comes from the **combination** of what pip picked and what's already on the box. That box is often shared, with other deployments pinned to its CUDA version, so "just upgrade the driver" isn't an answer.

Fixing it by hand means cross-referencing engine release notes, the PyTorch build matrix and per-CUDA wheel indexes. `env-doctor` does that for you in one command.

## How resolution works

An inference engine is a chain of pinned dependencies, and the whole chain has to run on your driver:

```
engine version  →  pinned torch  →  CUDA build of the wheels  →  kernel libs
 (vllm / sglang)     (e.g. 2.13)       (cu129 / cu130)           (flashinfer / sgl-kernel)
                                              ↓
                          must run on the INSTALLED NVIDIA driver
```

### The driver decides, not the toolkit

pip wheels bundle their own CUDA runtime. What decides whether they run is the **maximum CUDA version your driver supports**, not your system CUDA toolkit (`nvcc`). The toolkit only matters for kernels compiled at runtime (e.g. FlashInfer JIT), so a mismatch there is reported as a soft warning.

### CUDA minor-version compatibility

| Wheel's CUDA build vs. driver | Verdict |
|---|---|
| Build ≤ driver's max CUDA | ✅ Runs |
| Same major, newer minor (e.g. 12.9 build on a 12.6 driver) | ⚠️ Runs via [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) (driver ≥ 525 for 12.x), with a warning |
| Newer major (e.g. 13.0 build on a 12.x driver) | ❌ Fails |

This matters more than it looks: **vLLM doesn't publish CUDA 12.6 or 12.8 builds at all**, only CUDA 12.9 and 13.0. On a CUDA 12.6 box, the working answer is the 12.9 build, through minor-version compatibility.

### Fixes are ranked to avoid touching your box

1. **Same engine version, different CUDA build.** For example vLLM's per-version CUDA 12.9 wheel index, or SGLang's CUDA 12 install path.
2. **Newest engine version with a build that runs** on your driver. For example, SGLang 0.5.19 is the last release with a CUDA 12 path.
3. **Driver upgrade.** Offered last and flagged **⚠ RISKY**, because on a shared box it can break other deployments.

## Examples

### vLLM on a CUDA 12.6 box

```console
$ env-doctor install vllm
⚠️   🚀 vLLM 0.30.0  →  needs attention on driver CUDA 12.6
    Default wheel: CUDA 13.0, torch 2.13.0
    Pinned kernel libs: flashinfer-python 0.6.18.post1
    ❌ Default `pip install vllm==0.30.0` pulls a CUDA 13.0 build (torch 2.13.0), but the driver supports up to CUDA 12.6.
    ⚠️  CUDA 12.9 build runs on a CUDA 12.6 driver via CUDA minor-version compatibility (driver ≥ 525). ...
    → Fix options (best first):
      1. Install the CUDA 12.9 build of vLLM 0.30.0 — via CUDA minor-version compatibility
           uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129
      2. Upgrade the NVIDIA driver to ≥ 580 to use the default CUDA 13.0 build — on a shared box this can break other deployments  ⚠ RISKY
    📋 Copy-to-fix: uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129
```

You stay on the newest vLLM: only the CUDA build changes, and the driver stays as it is.

### SGLang: falling back to the last CUDA 12 release

SGLang 0.5.20 and later are CUDA 13 only. Its CUDA 12 path is a multi-step reinstall of co-pinned packages, and `env-doctor` generates every step:

```console
$ env-doctor install sglang
    → Fix options (best first):
      1. Use SGLang 0.5.19 — newest release with a build for this driver (CUDA 12.9) ... Last release with a CUDA 12 lane.
           uv pip install sglang==0.5.19
           uv pip install --force-reinstall torch==2.13.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129
           uv pip install --force-reinstall sglang-kernel==0.4.6.post1 --index-url https://docs.sglang.ai/whl/cu129/
           uv pip install --force-reinstall sgl-deep-gemm==0.1.7 --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
```

### An installed engine that will fail at runtime

Plain `env-doctor check` checks any installed vLLM/SGLang automatically. It reads versions from package metadata and never imports the engine, so the check stays fast:

```console
$ env-doctor check
❌  🚀 vLLM 0.30.0  →  incompatible with driver (CUDA 12.2)
    Installed: vLLM 0.30.0 (CUDA 13.0 build)
    ❌ Installed vLLM 0.30.0 is built for CUDA 13.0, but the driver supports up to CUDA 12.2 — it will fail at runtime.
    ⚠️  flashinfer-python 0.6.12 is installed but vLLM 0.30.0 pins 0.6.18.post1 — mismatched kernel libs can fail at import or produce wrong results.
    📋 Copy-to-fix: uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --torch-backend=cu129
```

It also catches **kernel-lib drift**: a FlashInfer or sgl-kernel version that doesn't match the version the engine pins.

## Usage reference

| Command | What it does |
|---|---|
| `env-doctor install vllm` | Install command for the newest vLLM that runs on your driver |
| `env-doctor install vllm@0.19.1` | A specific version (`vllm==0.19.1` also works) |
| `env-doctor install sglang --execute` | Run the recommended commands in order, stopping at the first failure. Never runs a driver-upgrade option. |
| `env-doctor check --engine vllm` | Diagnose before installing (repeatable: `--engine vllm --engine sglang@0.5.19`) |
| `env-doctor check --engine vllm --json` | Machine-readable: results under `checks.engines.vllm` |
| `env-doctor check --engine vllm --format html` | HTML report with an "Inference Engine" section |

Engine statuses feed the overall `check` status and exit code: an engine that will fail at runtime is an error (exit 2), and one that needs a non-default install is a warning (exit 1).

Install commands use [`uv`](https://docs.astral.sh/uv/), which is needed for `--torch-backend`. If `uv` isn't on your PATH, `pip install uv` is added as the first step.

### Python API and MCP

```python
from env_doctor import check
check(engines=["vllm", "sglang@0.5.19"])
```

AI assistants can use the [MCP server](mcp-integration.md) tools `engine_check` and `install_command` (`library="vllm"`).

## Which version does it pick?

With no version specified, `env-doctor` evaluates, in order:

1. **The version you asked for**: `vllm@0.19.1`
2. **The installed version** (`check` only)
3. **The newest version in env-doctor's compatibility table**

"Newest" means **newest verified in the table**, not necessarily the newest release on PyPI. Only versions whose CUDA builds and install paths have been checked get recommended. vLLM and SGLang release every 1–2 weeks, so a brand-new release can take a short while to appear. You can always ask for a specific version, and a version that isn't in the table is still evaluated with a warning.

## Where the data comes from

The compatibility table ([`inference_engines.json`](https://github.com/mitulgarg/env-doctor/blob/main/src/env_doctor/data/inference_engines.json)) is hand-curated and verified against:

- **Torch and kernel-lib pins:** each release's dependencies as published on PyPI
- **vLLM CUDA builds:** GitHub release assets and the `wheels.vllm.ai/<version>/<cuda>` indexes
- **SGLang CUDA 12 path:** SGLang's install docs and its `docs.sglang.ai/whl/cu129` index

The table updates from GitHub automatically (cached for 24 hours), so new engine releases reach you **without upgrading env-doctor**. Offline, the copy bundled with the package is used.

**Coverage:** vLLM 0.17.1 → 0.30.0 and SGLang 0.5.9 → 0.5.21, NVIDIA GPUs only. ROCm, XPU and Metal are not covered yet.

## See also

- [`check`](../commands/check.md#inference-engines-vllm-sglang): full environment diagnosis
- [`install`](../commands/install.md#inference-engines-vllm-sglang): safe install commands
- [MCP Integration](mcp-integration.md): `engine_check` tool
