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

    elif name == "dockerfile_validate":
        content = arguments.get("content", "")
        result = tools.dockerfile_validate(content)

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
