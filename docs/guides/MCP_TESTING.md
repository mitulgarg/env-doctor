# MCP Tools Testing Guide

This guide explains how to test all 11 env-doctor MCP tools using JSON-RPC.

## Available MCP Tools

| # | Tool Name | Description |
|---|-----------|-------------|
| 1 | `env_check` | Full GPU/CUDA environment diagnostics |
| 2 | `env_check_component` | Check specific component (driver, CUDA, cuDNN, etc.) |
| 3 | `python_compat_check` | Check Python version compatibility with installed AI libraries |
| 4 | `cuda_info` | Detailed CUDA toolkit information |
| 5 | `cudnn_info` | Detailed cuDNN library information |
| 6 | `cuda_install` | Step-by-step CUDA installation instructions |
| 7 | `install_command` | Get safe pip install command for AI libraries |
| 8 | `model_check` | Check if AI model fits on GPU |
| 9 | `model_list` | List all available AI models in database |
| 10 | `dockerfile_validate` | Validate Dockerfile for GPU config issues |
| 11 | `docker_compose_validate` | Validate docker-compose.yml for GPU config |

## Testing Methods

### Method 1: Automated Test Suite (Recommended)

Run all 11 tools automatically:

```bash
python tests/test_mcp_tools.py
```

This will:
- Connect to the MCP server
- Test each tool with sample inputs
- Display results in a structured format
- Show success/failure for each tool

**Sample Output:**
```
================================================================================
TESTING ENV-DOCTOR MCP TOOLS
================================================================================

[1/12] Listing available tools...
✓ Found 11 tools:
  - env_check: Run full GPU/CUDA environment diagnostics. Checks NVIDIA...
  - env_check_component: Run diagnostics for a specific component. Avail...
  - python_compat_check: Check Python version compatibility with installed AI...
  ...

[2/11] Testing env_check...
✓ env_check completed
  Status: warning
  Components checked: 5
  Components detected: 3

[3/11] Testing env_check_component (nvidia_driver)...
✓ env_check_component completed
  Status: success
  Detected: True
  Version: 535.129.03
```

### Method 2: Interactive Testing

Test tools interactively with custom inputs:

```bash
python tests/test_mcp_interactive.py
```

**Interactive Menu:**
```
Available tools:
   1. env_check               - Full environment diagnostics
   2. env_check_component     - Check specific component
   3. python_compat_check     - Python version compatibility check
   4. cuda_info               - Detailed CUDA toolkit info
   5. cudnn_info              - Detailed cuDNN info
   6. cuda_install            - CUDA installation guide
   7. install_command         - Get pip install command
   8. model_check             - Check if model fits on GPU
   9. model_list              - List available models
  10. dockerfile_validate     - Validate Dockerfile
  11. docker_compose_validate - Validate docker-compose.yml

Commands:
  - Enter tool number (1-11) to test a tool
  - Type 'list' to list tools again
  - Type 'all' to run all tools
  - Type 'quit' or 'exit' to exit
```

**Test Specific Tool:**
```bash
# Test cuda_install with auto-detection
python tests/test_mcp_interactive.py cuda_install

# Test install_command with custom args
python tests/test_mcp_interactive.py install_command '{"library":"tensorflow"}'

# Test model_check
python tests/test_mcp_interactive.py model_check '{"model_name":"llama-3-70b","precision":"int4"}'
```

### Method 3: Manual JSON-RPC Testing

For advanced users who want to see raw JSON-RPC protocol:

```bash
bash tests/test_mcp_manual.sh
```

Or manually with stdio:

```bash
# Start the MCP server
python -m env_doctor.mcp.server

# Send JSON-RPC requests (in another terminal or via pipe)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python -m env_doctor.mcp.server

echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"cuda_info","arguments":{}}}' | python -m env_doctor.mcp.server

echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"python_compat_check","arguments":{}}}' | python -m env_doctor.mcp.server
```

## Testing Individual Tools

### 1. env_check
```python
# Python
result = await session.call_tool("env_check", {})

# Expected response
{
  "components": {
    "nvidia_driver": {...},
    "cuda_toolkit": {...},
    "cudnn": {...}
  },
  "summary": {
    "status": "success",
    "component_count": 5,
    "detected_count": 3
  }
}
```

### 2. env_check_component
```python
# Check CUDA toolkit
result = await session.call_tool("env_check_component", {
    "component": "cuda_toolkit"
})

# Check NVIDIA driver
result = await session.call_tool("env_check_component", {
    "component": "nvidia_driver"
})
```

### 3. python_compat_check
```python
# Check Python version compatibility with installed AI libraries
result = await session.call_tool("python_compat_check", {})

# Expected response
{
  "detected": true,
  "python_version": "3.13.0",
  "libraries_checked": 2,
  "conflicts": [
    {
      "library": "torch",
      "installed_version": "2.5.1",
      "issue": "torch supports Python <=3.12, but you have Python 3.13",
      "note": "PyTorch 2.x supports Python 3.9-3.12. Python 3.13 support experimental."
    },
    {
      "library": "tensorflow",
      "installed_version": "2.15.0",
      "issue": "tensorflow supports Python <=3.12, but you have Python 3.13",
      "note": "TensorFlow 2.15+ requires Python 3.9-3.12. Python 3.13 not yet supported."
    }
  ],
  "cascades": [
    {
      "library": "torch",
      "severity": "high",
      "description": "PyTorch's Python version constraint affects all torch ecosystem packages",
      "affected": ["torchvision", "torchaudio", "triton"]
    }
  ],
  "issues": [
    "torch supports Python <=3.12, but you have Python 3.13",
    "tensorflow supports Python <=3.12, but you have Python 3.13"
  ],
  "recommendations": [
    "Consider using Python 3.12 or lower for full compatibility",
    "Cascade: torch constraint also affects: torchvision, torchaudio, triton"
  ]
}
```

### 4. cuda_info
```python
# Get detailed CUDA info
result = await session.call_tool("cuda_info", {})

# Response includes
{
  "detected": true,
  "version": "12.1",
  "metadata": {
    "installations": [...],
    "nvcc": {...},
    "cuda_home": {...},
    "driver_compatibility": {...}
  }
}
```

### 5. cudnn_info
```python
# Get detailed cuDNN info
result = await session.call_tool("cudnn_info", {})

# Response
{
  "detected": true,
  "version": "8.9.0",
  "path": "/usr/lib/x86_64-linux-gnu/libcudnn.so.8"
}
```

### 6. cuda_install
```python
# Auto-detect best CUDA version from driver
result = await session.call_tool("cuda_install", {})

# Specify CUDA version
result = await session.call_tool("cuda_install", {
    "version": "12.4"
})

# Response includes
{
  "platform": {
    "os": "linux",
    "distro": "ubuntu",
    "distro_version": "22.04",
    "arch": "x86_64",
    "is_wsl2": false
  },
  "recommended_version": "12.4",
  "install_info": {
    "label": "Ubuntu 22.04 (x86_64) - Network Install",
    "steps": [...],
    "post_install": [...],
    "verify": "nvcc --version"
  }
}
```

### 7. install_command
```python
# Get install command for PyTorch
result = await session.call_tool("install_command", {
    "library": "torch"
})

# Get install command for TensorFlow
result = await session.call_tool("install_command", {
    "library": "tensorflow"
})

# Response
{
  "library": "torch",
  "driver_detected": true,
  "driver_version": "535.129.03",
  "max_cuda": "12.2",
  "install_command": "pip install torch==2.5.1 torchvision==0.20.1..."
}
```

### 8. model_check
```python
# Check if llama-3-8b fits
result = await session.call_tool("model_check", {
    "model_name": "llama-3-8b"
})

# Check specific precision
result = await session.call_tool("model_check", {
    "model_name": "stable-diffusion-xl",
    "precision": "fp16"
})

# Response
{
  "success": true,
  "model_name": "llama-3-8b",
  "gpu_info": {...},
  "vram_requirements": {
    "fp16": {"vram_mb": 19200},
    "int4": {"vram_mb": 4800}
  },
  "compatibility": {...}
}
```

### 9. model_list
```python
# List all models
result = await session.call_tool("model_list", {})

# Response
{
  "models_by_category": {
    "llm": [...],
    "diffusion": [...],
    "audio": [...]
  },
  "stats": {
    "total_models": 60,
    "categories": 4
  }
}
```

### 10. dockerfile_validate
```python
# Validate Dockerfile content
dockerfile = """
FROM python:3.10
RUN pip install torch torchvision torchaudio
CMD ["python", "app.py"]
"""

result = await session.call_tool("dockerfile_validate", {
    "content": dockerfile
})

# Response
{
  "success": false,
  "error_count": 1,
  "warning_count": 2,
  "issues": [
    {
      "line_number": 2,
      "severity": "error",
      "issue": "Missing --index-url for PyTorch",
      "recommendation": "Add CUDA-specific index URL"
    }
  ]
}
```

### 11. docker_compose_validate
```python
# Validate docker-compose.yml
compose = """
version: '3.8'
services:
  app:
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: python train.py
"""

result = await session.call_tool("docker_compose_validate", {
    "content": compose
})

# Response
{
  "success": false,
  "error_count": 1,
  "warning_count": 0,
  "issues": [
    {
      "severity": "error",
      "issue": "Missing GPU device configuration",
      "recommendation": "Add deploy.resources.reservations.devices"
    }
  ]
}
```

## Integration with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "python",
      "args": ["-m", "env_doctor.mcp.server"]
    }
  }
}
```

Or if installed globally:
```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "env-doctor-mcp"
    }
  }
}
```

## Common Test Scenarios

### Scenario 1: New GPU Setup
```python
# 1. Check driver
await session.call_tool("env_check_component", {"component": "nvidia_driver"})

# 2. Check CUDA
await session.call_tool("cuda_info", {})

# 3. Get CUDA install instructions
await session.call_tool("cuda_install", {})

# 4. Get PyTorch install command
await session.call_tool("install_command", {"library": "torch"})
```

### Scenario 2: Model Deployment
```python
# 1. Check hardware
await session.call_tool("env_check", {})

# 2. Check if model fits
await session.call_tool("model_check", {"model_name": "llama-3-70b"})

# 3. Validate Dockerfile
await session.call_tool("dockerfile_validate", {"content": dockerfile})

# 4. Validate docker-compose
await session.call_tool("docker_compose_validate", {"content": compose})
```

### Scenario 3: Python Version Compatibility Check
```python
# 1. Check Python version against installed libraries
await session.call_tool("python_compat_check", {})

# 2. If conflicts found, get the safe install command for a compatible version
await session.call_tool("install_command", {"library": "torch"})
```

### Scenario 4: Debugging Installation Issues
```python
# 1. Full environment check
await session.call_tool("env_check", {})

# 2. Detailed CUDA analysis
await session.call_tool("cuda_info", {})

# 3. Detailed cuDNN analysis
await session.call_tool("cudnn_info", {})

# 4. Get correct install commands
await session.call_tool("install_command", {"library": "torch"})
```

## Troubleshooting

### Server Not Starting
```bash
# Check if env-doctor is installed
python -m env_doctor.mcp.server

# Check dependencies
pip list | grep mcp
```

### Tool Errors
Check the error message in the response:
```json
{
  "error": "No NVIDIA driver detected",
  "driver_download": "https://www.nvidia.com/Download/index.aspx"
}
```

### Invalid Arguments
Each tool has specific required/optional arguments. Check the tool definition:
```python
tools = await session.list_tools()
for tool in tools.tools:
    print(f"{tool.name}: {tool.inputSchema}")
```

## Performance Notes

- `env_check` scans all components (~1-2 seconds)
- `python_compat_check` scans installed libraries against compatibility matrix (~0.3 seconds)
- `cuda_info` and `cudnn_info` are fast (<0.5 seconds)
- `model_check` may fetch from HuggingFace API first time (cached afterward)
- `dockerfile_validate` and `docker_compose_validate` are instant
- `cuda_install` detects platform and looks up instructions (~0.2 seconds)
