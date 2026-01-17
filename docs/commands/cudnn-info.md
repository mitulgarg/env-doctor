# cudnn-info

Detect and validate cuDNN library installations.

## Usage

```bash
env-doctor cudnn-info
```

## What It Shows

### cuDNN Detection

- cuDNN version and build information
- Library file locations
- Header file locations

### Installation Validation

- Multiple installation detection
- Symlink validation (Linux)
- PATH configuration (Windows)

### Compatibility

- CUDA version compatibility
- Version requirements for popular frameworks

## Example Output

```
🧠 cuDNN ANALYSIS
============================================================

📚 cuDNN Installation:

  Version: 8.9.7
  CUDA Version: 12.x

  Libraries:
    ✅ libcudnn.so.8.9.7
    └─ Path: /usr/local/cuda-12.1/lib64/libcudnn.so.8

  Headers:
    ✅ cudnn.h
    └─ Path: /usr/local/cuda-12.1/include/cudnn.h

🔗 Symlinks (Linux):

  libcudnn.so → libcudnn.so.8 → libcudnn.so.8.9.7
  ✅ Symlink chain is valid

🎮 Compatibility:

  ✅ cuDNN 8.9.7 is compatible with CUDA 12.1
  ✅ Meets PyTorch 2.1+ requirements (cuDNN 8.5+)
  ✅ Meets TensorFlow 2.15+ requirements (cuDNN 8.6+)
```

## Multiple Installations

```
⚠️  Multiple cuDNN installations detected

  /usr/local/cuda-12.1/lib64/libcudnn.so.8.9.7
  /usr/lib/x86_64-linux-gnu/libcudnn.so.8.6.0

💡 Recommendations:
  - Ensure LD_LIBRARY_PATH prioritizes the correct version
  - Consider removing older installations to avoid conflicts
```

## Common Issues

### Missing cuDNN

```
❌ cuDNN not found

💡 Installation:
  1. Download from https://developer.nvidia.com/cudnn
  2. Extract to /usr/local/cuda-12.1/
  3. Or install via package manager:
     sudo apt install libcudnn8 libcudnn8-dev
```

### Broken Symlinks

```
❌ Broken symlink detected

  libcudnn.so → libcudnn.so.8 (missing)

💡 Fix:
  cd /usr/local/cuda/lib64
  sudo ln -sf libcudnn.so.8.9.7 libcudnn.so.8
  sudo ln -sf libcudnn.so.8 libcudnn.so
```

### Version Mismatch

```
⚠️  cuDNN version may not be optimal

  Installed: cuDNN 8.2.0
  PyTorch 2.1 recommends: cuDNN 8.5+

💡 Consider upgrading cuDNN for better performance
```

---

## Advanced Options

### JSON Output

```bash
env-doctor cudnn-info --json
```

## See Also

- [cuda-info](cuda-info.md) - CUDA toolkit analysis
- [check](check.md) - Full environment diagnosis
