# MCP Server Integration

Env-Doctor includes a built-in [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that exposes its diagnostic capabilities as read-only tools for AI assistants like Claude Desktop, Zed, and other MCP-compatible clients.

## What is MCP?

The Model Context Protocol is an open standard that enables AI assistants to securely access external tools and data sources. The env-doctor MCP server allows AI assistants to:

- Diagnose GPU/CUDA environments
- Check AI model compatibility
- Validate Dockerfiles for GPU issues
- List available models and their requirements

All operations are **read-only** - the MCP server cannot modify your system.

## Quick Start

### 1. Install env-doctor

```bash
pip install env-doctor
```

### 2. Configure Claude Desktop

Add the env-doctor MCP server to your Claude Desktop configuration:

**macOS/Linux:**
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:**
Edit `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "env-doctor-mcp"
    }
  }
}
```

### 3. Restart Claude Desktop

After saving the configuration, restart Claude Desktop. The env-doctor tools will be available automatically.

## Available Tools

The MCP server exposes 10 diagnostic tools:

### `env_check`

Run full GPU/CUDA environment diagnostics.

**What it checks:**
- NVIDIA driver version and status
- CUDA toolkit installation and configuration
- cuDNN library detection
- Python AI libraries (PyTorch, TensorFlow, JAX)
- WSL2 GPU forwarding (Windows)

**Returns:** Status, versions, issues, and recommendations for each component.

**Example prompt:**
> "Check my GPU environment"

---

### `env_check_component`

Run diagnostics for a specific component.

**Parameters:**
- `component` (required): One of:
  - `nvidia_driver` - GPU driver detection
  - `cuda_toolkit` - CUDA installation analysis
  - `cudnn` - cuDNN library detection
  - `python_library` - AI library compatibility
  - `wsl2` - WSL2 GPU forwarding validation

**Returns:** Detailed status for the specified component.

**Example prompt:**
> "Check my CUDA toolkit configuration"

---

### `model_check`

Check if an AI model fits on available GPU hardware.

**Parameters:**
- `model_name` (required): Model name or HuggingFace ID
  - Examples: `llama-3-8b`, `meta-llama/Llama-2-7b-hf`, `stable-diffusion-xl`
- `precision` (optional): Specific precision to check
  - Options: `fp32`, `fp16`, `bf16`, `int8`, `int4`, `fp8`

**Returns:** VRAM requirements, compatibility analysis, and recommendations.

**Example prompts:**
> "Can I run Llama 3 8B on my GPU?"
> "Check if stable-diffusion-xl fits in fp16"

---

### `model_list`

List all available AI models in the database.

**Returns:** Models grouped by category (LLM, diffusion, audio, vision, multimodal) with parameter counts and HuggingFace IDs.

**Example prompt:**
> "What models are available in the database?"

---

### `dockerfile_validate`

Validate Dockerfile content for GPU/CUDA configuration issues.

**Parameters:**
- `content` (required): Dockerfile content as a string

**What it checks:**
- CPU-only base images with GPU libraries
- Missing PyTorch `--index-url` flags
- CUDA version mismatches
- Driver installations in containers (anti-pattern)
- Deprecated package usage
- Runtime vs devel base image mismatches

**Returns:** List of issues with line numbers, severity, and corrected commands.

**Example prompt:**
> "Validate this Dockerfile: [paste Dockerfile content]"

---

### `docker_compose_validate`

Validate docker-compose.yml content for GPU configuration issues.

**Parameters:**
- `content` (required): docker-compose.yml content as a string

**What it checks:**
- Missing GPU runtime configuration
- Incorrect device mappings
- Missing environment variables
- Volume mount issues
- GPU resource allocation

**Returns:** List of issues with severity levels and corrected configuration.

**Example prompt:**
> "Validate this docker-compose.yml: [paste content]"

---

### `install_command`

Get safe installation command for a library based on detected GPU driver.

**Parameters:**
- `library` (required): Library name (e.g., "torch", "tensorflow", "jax")

**Returns:**
- Detected driver version and max CUDA support
- Safe pip install command with correct CUDA version

**Example prompts:**
> "What's the safe install command for PyTorch?"
> "How do I install TensorFlow for my GPU?"

---

### `cuda_info`

Get detailed CUDA toolkit information and diagnostics.

**Returns:**
- All CUDA installations found on system
- nvcc compiler information
- CUDA_HOME environment variable status
- Runtime library locations
- PATH and LD_LIBRARY_PATH configuration
- Driver compatibility analysis
- Detected issues and recommendations

**Example prompt:**
> "Show me detailed CUDA toolkit information"

---

### `cudnn_info`

Get detailed cuDNN library information and diagnostics.

**Returns:**
- cuDNN version and installation path
- All cuDNN libraries found
- Symlink status (Linux)
- PATH configuration (Windows)
- CUDA compatibility analysis
- Detected issues and recommendations

**Example prompt:**
> "Check my cuDNN installation"

---

### `cuda_install`

Get step-by-step CUDA Toolkit installation instructions for your platform.

**Parameters:**
- `version` (optional): Specific CUDA version to install (auto-detects from driver if not specified)

**Returns:**
- Platform detection (OS, distro, architecture)
- Recommended CUDA version based on driver
- Platform-specific installation steps
- Official download links

**Example prompts:**
> "How do I install CUDA Toolkit?"
> "Show me CUDA 12.1 installation steps"

## Configuration Options

### Basic Configuration

Minimal setup for Claude Desktop:

```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "env-doctor-mcp"
    }
  }
}
```

### Python Virtual Environment

If env-doctor is installed in a virtual environment:

```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "/path/to/venv/bin/env-doctor-mcp"
    }
  }
}
```

**macOS/Linux:**
```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "/home/user/.venv/bin/env-doctor-mcp"
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "C:\\Users\\user\\.venv\\Scripts\\env-doctor-mcp.exe"
    }
  }
}
```

### Multiple Python Versions

If you have multiple Python versions, specify the full path:

```json
{
  "mcpServers": {
    "env-doctor": {
      "command": "python3.10",
      "args": ["-m", "env_doctor.mcp"]
    }
  }
}
```

## Usage Examples

### Diagnosing Environment Issues

**User:** My PyTorch isn't detecting my GPU. Can you help?

**Claude with env-doctor MCP:**
1. Runs `env_check` to diagnose the full environment
2. Identifies version mismatches
3. Provides specific fix commands

---

### Checking Model Compatibility

**User:** I want to run Llama 2 70B. Will it fit on my 2x RTX 3090?

**Claude with env-doctor MCP:**
1. Runs `model_check` for `llama-2-70b`
2. Analyzes VRAM requirements across precisions
3. Recommends optimal precision (e.g., int4) or multi-GPU setup

---

### Validating Dockerfiles

**User:** Here's my Dockerfile for a PyTorch project. Are there any issues?

```dockerfile
FROM python:3.10
RUN pip install torch torchvision
```

**Claude with env-doctor MCP:**
1. Runs `dockerfile_validate` on the content
2. Detects CPU-only base image
3. Identifies missing `--index-url` flag
4. Provides corrected Dockerfile with GPU-enabled base image

---

### Listing Available Models

**User:** What AI models can I check compatibility for?

**Claude with env-doctor MCP:**
1. Runs `model_list`
2. Shows models grouped by category (LLMs, diffusion, audio, etc.)
3. Includes parameter counts and HuggingFace IDs

## Supported MCP Clients

The env-doctor MCP server is compatible with:

- **Claude Desktop** (macOS, Windows, Linux)
- **Zed Editor** (via MCP support)
- **Any MCP-compatible client** using stdio transport

## Troubleshooting

### Server Not Appearing in Claude Desktop

1. **Verify installation:**
   ```bash
   env-doctor-mcp --help
   ```

2. **Check configuration file location:**
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

3. **Validate JSON syntax:**
   Use a JSON validator to ensure your config file is valid.

4. **Check Claude Desktop logs:**
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`

5. **Restart Claude Desktop** after making config changes.

### Permission Errors

If you see permission errors, ensure env-doctor-mcp is executable:

```bash
# macOS/Linux
chmod +x $(which env-doctor-mcp)
```

### Virtual Environment Issues

If using a virtual environment, use the full path to the env-doctor-mcp executable:

```bash
# Find the path
which env-doctor-mcp

# Or
python -c "import sys; print(sys.prefix + '/bin/env-doctor-mcp')"
```

## Security Considerations

The env-doctor MCP server is designed with security in mind:

- **Read-only operations:** Cannot modify system configuration
- **No network access:** Doesn't make external API calls (except optional HuggingFace model lookups)
- **Local execution:** Runs entirely on your machine
- **No data collection:** Doesn't send diagnostic data anywhere

The only write operation is caching HuggingFace model metadata locally to `~/.env_doctor_cache.json`.

## Advanced Usage

### Programmatic Access

You can also use the MCP server programmatically in Python:

```python
from env_doctor.mcp import tools

# Check full environment
result = tools.env_check()
print(result)

# Check specific component
result = tools.env_check_component("nvidia_driver")
print(result)

# Check model compatibility
result = tools.model_check("llama-3-8b")
print(result)

# List available models
result = tools.model_list()
print(result)

# Validate Dockerfile
dockerfile_content = """
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
"""
result = tools.dockerfile_validate(dockerfile_content)
print(result)
```

### Custom MCP Clients

To integrate env-doctor with a custom MCP client:

```python
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to env-doctor MCP server
server_params = StdioServerParameters(
    command="env-doctor-mcp",
    args=[]
)

async with stdio_client(server_params) as (read, write):
    # Use the client
    pass
```

## See Also

- [Model Context Protocol Documentation](https://modelcontextprotocol.io)
- [Claude Desktop Setup](https://claude.ai/desktop)
- [check Command](../commands/check.md) - CLI equivalent of env_check
- [model Command](../commands/model.md) - CLI equivalent of model_check
- [dockerfile Command](../commands/dockerfile.md) - CLI equivalent of dockerfile_validate
