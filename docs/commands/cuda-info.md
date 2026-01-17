# cuda-info

Get detailed information about your CUDA toolkit installations.

## Usage

```bash
env-doctor cuda-info
```

## What It Shows

### CUDA Installations

- All CUDA toolkit installations found on your system
- Version of each installation
- Installation paths

### Environment Variables

- `CUDA_HOME` configuration
- `PATH` entries for CUDA binaries
- `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows) for libraries

### Runtime Libraries

- `libcudart` (CUDA runtime) status
- Library paths and versions

### Compatibility

- Driver compatibility with installed toolkit(s)
- Potential version conflicts

## Example Output

```
🔧 CUDA TOOLKIT ANALYSIS
============================================================

📍 Installations Found:

  CUDA 12.1.1
  └─ Path: /usr/local/cuda-12.1
  └─ nvcc: /usr/local/cuda-12.1/bin/nvcc

  CUDA 11.8.0
  └─ Path: /usr/local/cuda-11.8
  └─ nvcc: /usr/local/cuda-11.8/bin/nvcc

🌐 Environment Configuration:

  CUDA_HOME: /usr/local/cuda-12.1
  PATH:
    ✅ /usr/local/cuda-12.1/bin
  LD_LIBRARY_PATH:
    ✅ /usr/local/cuda-12.1/lib64

📚 Runtime Libraries:

  libcudart.so.12.1.105
  └─ Path: /usr/local/cuda-12.1/lib64/libcudart.so.12

🎮 Driver Compatibility:

  Driver: 535.146.02 (supports up to CUDA 12.2)
  ✅ CUDA 12.1 is compatible with your driver
```

## Multiple Installations

When multiple CUDA versions are installed:

```
⚠️  Multiple CUDA installations detected

  Active: CUDA 12.1 (via CUDA_HOME)
  Also found: CUDA 11.8

💡 Recommendations:
  - Ensure CUDA_HOME points to the version you need
  - Check PATH order if using nvcc
```

## Common Issues

### CUDA_HOME Not Set

```
❌ CUDA_HOME is not set

💡 Fix:
  export CUDA_HOME=/usr/local/cuda-12.1
  export PATH=$CUDA_HOME/bin:$PATH
```

### PATH Mismatch

```
⚠️  PATH contains different CUDA version than CUDA_HOME

  CUDA_HOME: /usr/local/cuda-12.1
  PATH nvcc: /usr/local/cuda-11.8/bin/nvcc

💡 Fix:
  Ensure PATH includes $CUDA_HOME/bin before other CUDA paths
```

---

## Advanced Options

### JSON Output

```bash
env-doctor cuda-info --json
```

## See Also

- [cudnn-info](cudnn-info.md) - cuDNN library analysis
- [check](check.md) - Full environment diagnosis
- [debug](debug.md) - Verbose detector output
