# dockerfile

Validate Dockerfiles for GPU/CUDA configuration issues before building.

## Usage

```bash
env-doctor dockerfile [PATH]
```

If no path is provided, looks for `Dockerfile` in the current directory.

## What It Validates

### Base Images

- Detects CPU-only images (`python:3.10`, `ubuntu:22.04`)
- Provides DB-driven GPU base image recommendations
- Suggests appropriate runtime vs devel images

### PyTorch Installation

- Ensures `pip install torch` has correct `--index-url`
- Validates version compatibility with base image CUDA
- Uses verified install commands from the database

### Library Compatibility

- Validates pinned versions against DB-verified combinations
- Checks multi-library compatibility (torch + tensorflow + jax)
- Flags deprecated packages (`tensorflow-gpu`)

### Build Requirements

- Detects compilation requirements (flash-attn, xformers)
- Enforces `-devel` base images when needed
- Warns about unnecessary toolkit installs

### Common Mistakes

- Flags NVIDIA driver installs (must be on host)
- Warns about bloating images with unnecessary packages

## Example Output

```
🐳  DOCKERFILE VALIDATION: Dockerfile

❌  ERRORS (2):
------------------------------------------------------------

Line 1:
  Issue: CPU-only base image detected: python:3.10
  Fix:   Use a GPU-enabled base image

  Suggested fix:
    FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
    # Or: FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    # Or: FROM tensorflow/tensorflow:latest-gpu

Line 8:
  Issue: PyTorch installation missing --index-url flag
  Fix:   Add --index-url to install the correct CUDA version

  Suggested fix:
    RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

⚠️   WARNINGS (1):
------------------------------------------------------------

Line 15:
  Issue: Installing CUDA toolkit in container
  Fix:   Use a CUDA base image instead to reduce image size

SUMMARY:
  ❌ Errors:   2
  ⚠️  Warnings: 1
  ℹ️  Info:     0
```

## Validation Rules

### Base Image Rules

| Image Type | Recommendation |
|------------|----------------|
| `python:*` | Use `nvidia/cuda:*` or `pytorch/pytorch:*` |
| `ubuntu:*` | Use `nvidia/cuda:*-base-ubuntu*` |
| `nvidia/cuda:*-runtime-*` | Good for inference |
| `nvidia/cuda:*-devel-*` | Required for compilation |

### PyTorch Install Rules

```dockerfile
# ❌ Wrong - gets whatever CUDA version pip decides
RUN pip install torch

# ✅ Correct - explicit CUDA version
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Compilation Detection

When these packages are detected, `-devel` base image is required:

- `flash-attn`
- `xformers` (when building from source)
- Any package with `--no-binary`

## See Also

- [docker-compose](docker-compose.md) - Validate docker-compose.yml
- [Docker Validation Guide](../guides/docker-validation.md) - Full guide
