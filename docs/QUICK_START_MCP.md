# Quick Start: MCP Testing

## Run All Tests (Fast)
```bash
python tests/test_mcp_tools.py
```

## Interactive Testing
```bash
python tests/test_mcp_interactive.py
```

## Test Specific Tool
```bash
# CUDA installation guide
python tests/test_mcp_interactive.py cuda_install

# Get PyTorch install command
python tests/test_mcp_interactive.py install_command '{"library":"torch"}'

# Check if model fits
python tests/test_mcp_interactive.py model_check '{"model_name":"llama-3-8b"}'

# Full environment check
python tests/test_mcp_interactive.py env_check

# Detailed CUDA info
python tests/test_mcp_interactive.py cuda_info
```

## All 10 Tools

| Tool | Quick Test Command |
|------|-------------------|
| `env_check` | `python tests/test_mcp_interactive.py env_check` |
| `env_check_component` | `python tests/test_mcp_interactive.py env_check_component '{"component":"nvidia_driver"}'` |
| `cuda_info` | `python tests/test_mcp_interactive.py cuda_info` |
| `cudnn_info` | `python tests/test_mcp_interactive.py cudnn_info` |
| `cuda_install` | `python tests/test_mcp_interactive.py cuda_install` |
| `install_command` | `python tests/test_mcp_interactive.py install_command '{"library":"torch"}'` |
| `model_check` | `python tests/test_mcp_interactive.py model_check '{"model_name":"llama-3-8b"}'` |
| `model_list` | `python tests/test_mcp_interactive.py model_list` |
| `dockerfile_validate` | (needs content string) |
| `docker_compose_validate` | (needs content string) |

## Example Session
```bash
# 1. Check what's installed
python tests/test_mcp_interactive.py env_check

# 2. Get CUDA installation instructions
python tests/test_mcp_interactive.py cuda_install

# 3. Get PyTorch install command
python tests/test_mcp_interactive.py install_command '{"library":"torch"}'

# 4. Check if your target model will fit
python tests/test_mcp_interactive.py model_check '{"model_name":"llama-3-70b"}'
```

## Output Format
All tools return JSON with consistent structure:
```json
{
  "success": true/false,
  "detected": true/false,
  "version": "...",
  "status": "success/warning/error",
  "issues": [...],
  "recommendations": [...],
  "error": "..." (if failed)
}
```

## Common Use Cases

### Setup New Machine
1. `env_check` - see what's missing
2. `cuda_install` - get installation steps
3. `install_command` - get library commands

### Debug Issues
1. `cuda_info` - detailed CUDA analysis
2. `cudnn_info` - detailed cuDNN analysis
3. `env_check` - overall status

### Deploy Model
1. `model_check` - will it fit?
2. `dockerfile_validate` - validate config
3. `docker_compose_validate` - validate deployment

## See Full Documentation
Read `MCP_TESTING.md` for:
- Detailed examples for each tool
- JSON-RPC protocol details
- Integration with Claude Desktop
- Troubleshooting guide
