# debug

Get detailed information from all detectors for troubleshooting.

## Usage

```bash
env-doctor debug
```

## What It Shows

### All Detector Results

- Raw output from every registered detector
- Detection status (success, warning, error, not_found)
- Version information

### Detection Metadata

- Internal detection methods used
- File paths examined
- Detailed status information

### Registry Information

- List of all available detectors
- Detector execution order

### Error Details

- Full exception traces
- Diagnostic information for failures

## Example Output

```
🔍 DEBUG MODE - Detailed Detector Information
============================================================
Registered Detectors: cuda_toolkit, nvidia_driver, python_library, wsl2, cudnn

--- WSL2 ---
Status: Status.SUCCESS
Component: wsl2
Version: wsl2
Metadata: {
  'environment': 'WSL2',
  'gpu_forwarding': 'enabled',
  'wsl_lib_path': '/usr/lib/wsl/lib',
  'libcuda_found': True
}

--- NVIDIA DRIVER ---
Status: Status.SUCCESS
Component: nvidia_driver
Version: 535.146.02
Path: /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
Metadata: {
  'max_cuda_version': '12.2',
  'detection_method': 'nvml',
  'gpu_name': 'NVIDIA GeForce RTX 3090',
  'gpu_memory': 24576
}

--- CUDA TOOLKIT ---
Status: Status.SUCCESS
Component: cuda_toolkit
Version: 12.1.1
Path: /usr/local/cuda-12.1
Metadata: {
  'nvcc_path': '/usr/local/cuda-12.1/bin/nvcc',
  'cuda_home': '/usr/local/cuda-12.1',
  'multiple_installations': ['/usr/local/cuda-12.1', '/usr/local/cuda-11.8']
}

--- CUDNN ---
Status: Status.SUCCESS
Component: cudnn
Version: 8.9.7
Path: /usr/local/cuda-12.1/lib64/libcudnn.so.8
Metadata: {
  'cuda_version': '12.x',
  'header_path': '/usr/local/cuda-12.1/include/cudnn.h'
}

--- PYTHON LIBRARIES ---
Status: Status.SUCCESS
Component: python_library
Libraries: {
  'torch': {
    'version': '2.1.0+cu121',
    'cuda_version': '12.1',
    'detection_method': 'import'
  },
  'tensorflow': {
    'status': 'not_installed'
  }
}
```

## When to Use Debug Mode

### Unexpected Results

When `env-doctor check` shows results that don't match your expectations:

```bash
# Regular check shows CUDA not found, but you know it's installed
env-doctor check
# Shows: ⚠️ No system CUDA toolkit found

# Debug to see what happened
env-doctor debug
# Shows: Detection attempted at /usr/local/cuda (symlink broken)
```

### Contributing or Reporting Issues

Include debug output when:

- Reporting bugs
- Contributing new detectors
- Validating behavior changes

### Understanding Detection Methods

Debug shows how each component was detected:

- NVML for driver detection
- File system search for CUDA
- Python imports for libraries

## See Also

- [check](check.md) - Standard environment check
- [Architecture](../architecture.md) - How detectors work
