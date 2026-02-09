"""
MCP (Model Context Protocol) server for env-doctor.

Exposes env-doctor's diagnostic capabilities as read-only MCP tools
for AI assistants like Claude Desktop.

Usage:
    # Run the MCP server (stdio transport)
    env-doctor-mcp

    # Or programmatically
    from env_doctor.mcp import serve
    serve()
"""

from .server import serve

__all__ = ["serve"]
