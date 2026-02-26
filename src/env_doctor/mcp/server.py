"""
MCP server for env-doctor.

Provides a stdio-based MCP server that exposes env-doctor's
diagnostic capabilities as read-only tools.
"""

import asyncio
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from . import tools

# Create the MCP server instance
server = Server("env-doctor")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available env-doctor MCP tools."""
    return [
        Tool(
            name="env_check",
            description=(
                "Run full GPU/CUDA environment diagnostics. "
                "Checks NVIDIA driver, CUDA toolkit, cuDNN, Python AI libraries (PyTorch, TensorFlow, JAX), "
                "and WSL2 configuration. Returns status, versions, and recommendations for each component."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="env_check_component",
            description=(
                "Run diagnostics for a specific component. "
                "Available components: nvidia_driver, cuda_toolkit, cudnn, python_library, wsl2."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": "Component to check (nvidia_driver, cuda_toolkit, cudnn, python_library, wsl2)",
                        "enum": ["nvidia_driver", "cuda_toolkit", "cudnn", "python_library", "wsl2"],
                    },
                },
                "required": ["component"],
            },
        ),
        Tool(
            name="model_check",
            description=(
                "Check if an AI model fits on available GPU hardware. "
                "Analyzes VRAM requirements across precisions (fp32, fp16, bf16, int8, int4) "
                "and provides recommendations for running the model."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name (e.g., 'llama-3-8b', 'meta-llama/Llama-2-7b-hf', 'stable-diffusion-xl')",
                    },
                    "precision": {
                        "type": "string",
                        "description": "Optional specific precision to check",
                        "enum": ["fp32", "fp16", "bf16", "int8", "int4", "fp8"],
                    },
                },
                "required": ["model_name"],
            },
        ),
        Tool(
            name="model_list",
            description=(
                "List all available AI models in the database, grouped by category "
                "(LLM, diffusion, audio, VLM). Includes parameter counts and HuggingFace IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="python_compat_check",
            description=(
                "Check Python version compatibility with installed AI libraries. "
                "Detects version conflicts where the current Python version is outside "
                "a library's supported range, and identifies dependency cascades where "
                "one library's constraint forces version limits on downstream packages."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="dockerfile_validate",
            description=(
                "Validate Dockerfile content for GPU/CUDA configuration issues. "
                "Checks for: CPU-only base images, missing PyTorch --index-url flags, "
                "CUDA version mismatches, driver installations in containers, "
                "and deprecated package usage."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Dockerfile content to validate",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="docker_compose_validate",
            description=(
                "Validate docker-compose.yml content for GPU configuration issues. "
                "Checks for: missing deploy.resources.reservations.devices, "
                "incorrect GPU driver settings, missing runtime: nvidia, "
                "and other GPU passthrough configuration problems."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "docker-compose.yml content to validate",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="install_command",
            description=(
                "Get the safe pip install command for an AI library based on the detected GPU driver. "
                "Automatically determines the correct CUDA version and wheel URL for libraries like "
                "PyTorch, TensorFlow, and JAX."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "library": {
                        "type": "string",
                        "description": "Library name (e.g., 'torch', 'tensorflow', 'jax')",
                    },
                },
                "required": ["library"],
            },
        ),
        Tool(
            name="cuda_info",
            description=(
                "Get detailed CUDA toolkit information including: nvcc version and path, "
                "all CUDA installations, CUDA_HOME configuration, PATH/LD_LIBRARY_PATH status, "
                "libcudart runtime library, and driver compatibility analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="cudnn_info",
            description=(
                "Get detailed cuDNN library information including: version, library paths, "
                "symlink status (Linux), PATH configuration (Windows), multiple version detection, "
                "and CUDA compatibility."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="wsl_info",
            description=(
                "Get detailed WSL (Windows Subsystem for Linux) environment information "
                "including: environment type (Native Linux / WSL1 / WSL2), kernel version, "
                "GPU forwarding status, and diagnostic checklist for WSL2 CUDA support."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="cuda_install",
            description=(
                "Get step-by-step CUDA Toolkit installation instructions tailored to the user's platform. "
                "Detects OS/distro, recommends the best CUDA version based on GPU driver, "
                "and provides copy-paste installation commands for Ubuntu, Debian, RHEL, Fedora, WSL2, Windows, and Conda."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "version": {
                        "type": "string",
                        "description": "Optional specific CUDA version to install (e.g., '12.6', '12.4', '12.1', '11.8'). If not specified, auto-detects best version from GPU driver.",
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls by dispatching to the appropriate function."""
    import json

    result: dict[str, Any]

    if name == "env_check":
        result = tools.env_check()

    elif name == "env_check_component":
        component = arguments.get("component", "")
        result = tools.env_check_component(component)

    elif name == "model_check":
        model_name = arguments.get("model_name", "")
        precision = arguments.get("precision")
        result = tools.model_check(model_name, precision)

    elif name == "model_list":
        result = tools.model_list()

    elif name == "python_compat_check":
        result = tools.python_compat_check()

    elif name == "dockerfile_validate":
        content = arguments.get("content", "")
        result = tools.dockerfile_validate(content)

    elif name == "docker_compose_validate":
        content = arguments.get("content", "")
        result = tools.docker_compose_validate(content)

    elif name == "install_command":
        library = arguments.get("library", "")
        result = tools.install_command(library)

    elif name == "cuda_info":
        result = tools.cuda_info()

    elif name == "cudnn_info":
        result = tools.cudnn_info()

    elif name == "wsl_info":
        result = tools.wsl_info()

    elif name == "cuda_install":
        version = arguments.get("version")
        result = tools.cuda_install(version)

    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


def serve() -> None:
    """
    Start the MCP server using stdio transport.

    This is the main entry point for the env-doctor-mcp command.
    """
    async def run_server():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run_server())


if __name__ == "__main__":
    serve()
