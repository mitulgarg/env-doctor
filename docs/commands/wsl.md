# wsl

Display comprehensive WSL environment information and GPU forwarding diagnostics.

## Usage

```bash
env-doctor wsl [--json]
```

## Options

| Option | Description |
|--------|-------------|
| `--json` | Output results in JSON format for programmatic consumption |

## Description

The `wsl` command runs a dedicated WSL environment analysis. It detects whether you are running inside WSL1, WSL2, or native Linux and — for WSL2 — validates the full GPU forwarding chain:

- Whether the internal NVIDIA driver (which breaks forwarding) is present
- Whether the WSL2 CUDA library (`/usr/lib/wsl/lib/libcuda.so`) exists
- Whether `nvidia-smi` is reachable from inside WSL2
- Overall GPU forwarding status

!!! note
    This command only runs on Linux. On macOS or Windows it will exit with an error.

## Example Output

### WSL2 with GPU forwarding enabled

```
============================================================
🐧  DETAILED WSL ANALYSIS
============================================================

📌  Environment Type: WSL2
    Kernel Version: 5.15.167.4-microsoft-standard-WSL2

🔍  GPU Forwarding Checklist:
    ✅ No internal NVIDIA driver (correct)
    ✅ WSL2 CUDA library found (/usr/lib/wsl/lib/libcuda.so)
    ✅ nvidia-smi working

✅  GPU Forwarding: Enabled
============================================================
```

### WSL2 with GPU forwarding broken

```
============================================================
🐧  DETAILED WSL ANALYSIS
============================================================

📌  Environment Type: WSL2
    Kernel Version: 5.15.167.4-microsoft-standard-WSL2

🔍  GPU Forwarding Checklist:
    ❌ Internal NVIDIA driver found (breaks GPU forwarding)
    ❌ WSL2 CUDA library missing (/usr/lib/wsl/lib/libcuda.so)
    ❌ nvidia-smi not working

❌  GPU Forwarding: Not working

⚠️   Issues Detected:
    • NVIDIA driver installed inside WSL2 — this breaks GPU forwarding

💡  Recommendations:
    → Remove the internal NVIDIA driver: sudo apt remove --purge nvidia-* libnvidia-*
    → Install the NVIDIA driver on Windows, not inside WSL2
============================================================
```

### WSL1

```
============================================================
🐧  DETAILED WSL ANALYSIS
============================================================

📌  Environment Type: WSL1

❌  WSL1 does not support CUDA/GPU computing
============================================================
```

### Native Linux

```
============================================================
🐧  DETAILED WSL ANALYSIS
============================================================

📌  Environment Type: Native Linux

✅  Native Linux (WSL not applicable)
============================================================
```

## JSON Output

```bash
env-doctor wsl --json
```

```json
{
  "component": "wsl2",
  "status": "success",
  "detected": true,
  "version": "wsl2",
  "metadata": {
    "environment": "WSL2",
    "kernel_version": "5.15.167.4-microsoft-standard-WSL2",
    "has_internal_driver": false,
    "has_libcuda": true,
    "cuda_lib_path": "/usr/lib/wsl/lib/libcuda.so",
    "nvidia_smi_works": true,
    "gpu_forwarding": "enabled"
  },
  "issues": [],
  "recommendations": []
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | WSL2 with GPU forwarding enabled, or native Linux |
| `1` | WSL2 GPU forwarding not working, WSL1 detected, or platform not supported |

## Difference from `check`

The main `check` command includes a brief WSL environment summary as part of its full diagnosis. `wsl` focuses exclusively on the WSL environment and provides a more detailed GPU forwarding checklist — useful for isolating WSL-specific issues without the overhead of a full system scan.

## Common Issues

### Internal NVIDIA driver installed inside WSL2

```
❌ Internal NVIDIA driver found (breaks GPU forwarding)
```

Remove the driver from inside WSL2 and rely on the Windows host driver instead:

```bash
sudo apt remove --purge nvidia-* libnvidia-*
sudo apt autoremove
```

Then restart WSL from PowerShell:

```powershell
wsl --shutdown
```

### Missing WSL2 CUDA library

```
❌ WSL2 CUDA library missing (/usr/lib/wsl/lib/libcuda.so)
```

This file is provided by the Windows NVIDIA driver. Update your Windows driver to version 470.76 or newer:

1. Download from [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Install on Windows and restart
3. Re-open WSL2 and run `env-doctor wsl` again

### `nvidia-smi` not working

Possible causes:

1. **Windows driver too old** — update to 470.76+
2. **WSL not updated** — run `wsl --update` in PowerShell
3. **Running WSL1** — upgrade with `wsl --set-version <distro> 2`

## See Also

- [WSL2 GPU Support Guide](../guides/wsl2.md) — Full setup and troubleshooting guide
- [check](check.md) — Full environment diagnosis (includes WSL summary)
- [cuda-info](cuda-info.md) — Detailed CUDA toolkit analysis
- [debug](debug.md) — Verbose detector output
