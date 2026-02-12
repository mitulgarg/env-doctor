# MCP Tools Testing

This directory contains test scripts for the env-doctor MCP (Model Context Protocol) server.

## Quick Start

```bash
# Run all MCP tools (automated)
python tests/test_mcp_tools.py

# Interactive testing
python tests/test_mcp_interactive.py

# Manual JSON-RPC testing
bash tests/test_mcp_manual.sh
```

## Test Files

### test_mcp_tools.py
**Automated test suite** - Runs all 10 MCP tools with sample inputs and displays results.

**Usage:**
```bash
python tests/test_mcp_tools.py
```

**Output:**
- Lists all available tools
- Tests each tool sequentially
- Shows success/failure status
- Displays key results from each tool

### test_mcp_interactive.py
**Interactive testing** - Menu-driven interface for testing individual tools.

**Usage:**
```bash
# Interactive menu
python tests/test_mcp_interactive.py

# Test specific tool
python tests/test_mcp_interactive.py cuda_install
python tests/test_mcp_interactive.py install_command '{"library":"torch"}'
```

**Features:**
- Menu of all 10 tools
- Customizable arguments
- Run all tools sequentially
- Individual tool testing

### test_mcp_manual.sh
**Raw JSON-RPC testing** - Shell script demonstrating manual JSON-RPC protocol.

**Usage:**
```bash
bash tests/test_mcp_manual.sh
```

**Purpose:**
- Shows raw JSON-RPC protocol
- Useful for debugging
- Low-level integration testing

## All 10 MCP Tools

| Tool | Description |
|------|-------------|
| `env_check` | Full environment diagnostics |
| `env_check_component` | Check specific component |
| `cuda_info` | Detailed CUDA toolkit info |
| `cudnn_info` | Detailed cuDNN info |
| `cuda_install` | CUDA installation guide |
| `install_command` | Get pip install command |
| `model_check` | Check if model fits on GPU |
| `model_list` | List available models |
| `dockerfile_validate` | Validate Dockerfile |
| `docker_compose_validate` | Validate docker-compose.yml |

## Documentation

See full documentation in:
- **Quick Start**: `docs/QUICK_START_MCP.md`
- **Complete Guide**: `docs/MCP_TESTING.md`

## Requirements

The MCP test scripts require:
- Python 3.10+
- `mcp` package installed
- env-doctor installed in development mode

**Install:**
```bash
pip install -e .
pip install mcp
```

## Example Output

```
================================================================================
TESTING ENV-DOCTOR MCP TOOLS
================================================================================

[1/11] Listing available tools...
✓ Found 10 tools:
  - env_check: Run full GPU/CUDA environment diagnostics...
  - cuda_install: Get step-by-step CUDA Toolkit installation...
  ...

[6/11] Testing cuda_install (auto-detect)...
✓ cuda_install completed
  Platform: linux - ubuntu 22.04
  Recommended CUDA: 12.6
  Steps: 4 installation steps
```

## Integration Testing

These tests verify that:
1. The MCP server starts correctly
2. All tools are registered and discoverable
3. Tools accept correct arguments
4. Tools return valid JSON responses
5. Error handling works properly

## CI/CD

To run MCP tests in CI:

```yaml
- name: Test MCP Tools
  run: |
    pip install mcp
    python tests/test_mcp_tools.py
```
