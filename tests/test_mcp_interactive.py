#!/usr/bin/env python3
"""
Interactive MCP tool tester for env-doctor.

Allows testing individual MCP tools with custom arguments.
"""
import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


TOOLS_INFO = {
    "env_check": {
        "description": "Full environment diagnostics",
        "args": {},
    },
    "env_check_component": {
        "description": "Check specific component",
        "args": {"component": "nvidia_driver"},
    },
    "cuda_info": {
        "description": "Detailed CUDA toolkit info",
        "args": {},
    },
    "cudnn_info": {
        "description": "Detailed cuDNN info",
        "args": {},
    },
    "cuda_install": {
        "description": "CUDA installation guide",
        "args": {},
        "args_optional": {"version": "12.4"},
    },
    "install_command": {
        "description": "Get pip install command",
        "args": {"library": "torch"},
    },
    "model_check": {
        "description": "Check if model fits on GPU",
        "args": {"model_name": "llama-3-8b"},
    },
    "model_list": {
        "description": "List available models",
        "args": {},
    },
    "dockerfile_validate": {
        "description": "Validate Dockerfile",
        "args": {"content": "FROM python:3.10\nRUN pip install torch\n"},
    },
    "docker_compose_validate": {
        "description": "Validate docker-compose.yml",
        "args": {"content": "version: '3.8'\nservices:\n  app:\n    image: pytorch/pytorch\n"},
    },
}


async def call_tool(session: ClientSession, tool_name: str, args: dict):
    """Call a tool and display results."""
    print(f"\n{'=' * 80}")
    print(f"CALLING: {tool_name}")
    print(f"ARGUMENTS: {json.dumps(args, indent=2)}")
    print(f"{'=' * 80}\n")

    try:
        result = await session.call_tool(tool_name, args)
        data = json.loads(result.content[0].text)

        print("RESULT:")
        print(json.dumps(data, indent=2))

        return data
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def interactive_test():
    """Interactive testing session."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "env_doctor.mcp.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 80)
            print("ENV-DOCTOR MCP INTERACTIVE TESTER")
            print("=" * 80)
            print("\nAvailable tools:")
            for i, (name, info) in enumerate(TOOLS_INFO.items(), 1):
                print(f"  {i:2d}. {name:25s} - {info['description']}")

            print("\nCommands:")
            print("  - Enter tool number (1-10) to test a tool")
            print("  - Type 'list' to list tools again")
            print("  - Type 'all' to run all tools")
            print("  - Type 'quit' or 'exit' to exit")

            while True:
                try:
                    print("\n" + "-" * 80)
                    choice = input("Enter command: ").strip().lower()

                    if choice in ("quit", "exit", "q"):
                        print("Goodbye!")
                        return  # Return instead of break to cleanly exit async context

                    elif choice == "list":
                        for i, (name, info) in enumerate(TOOLS_INFO.items(), 1):
                            print(f"  {i:2d}. {name:25s} - {info['description']}")

                    elif choice == "all":
                        print("\nRunning all tools...")
                        for name, info in TOOLS_INFO.items():
                            args = info["args"]
                            await call_tool(session, name, args)
                            input("\nPress Enter to continue...")

                    elif choice.isdigit():
                        idx = int(choice) - 1
                        tools_list = list(TOOLS_INFO.keys())

                        if 0 <= idx < len(tools_list):
                            tool_name = tools_list[idx]
                            info = TOOLS_INFO[tool_name]

                            # Use default args
                            args = info["args"].copy()

                            # Ask if user wants to customize args
                            if args:
                                print(f"\nDefault arguments: {json.dumps(args, indent=2)}")
                                customize = input("Customize arguments? (y/N): ").strip().lower()

                                if customize == 'y':
                                    for key in args.keys():
                                        value = input(f"  {key} [{args[key]}]: ").strip()
                                        if value:
                                            args[key] = value

                            await call_tool(session, tool_name, args)
                        else:
                            print(f"Invalid choice: {choice}")

                    else:
                        print(f"Unknown command: {choice}")

                except KeyboardInterrupt:
                    print("\n\nUse 'quit' to exit")
                except Exception as e:
                    print(f"Error: {e}")


async def test_specific_tool(tool_name: str, custom_args: dict = None):
    """Test a specific tool."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "env_doctor.mcp.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if tool_name not in TOOLS_INFO:
                print(f"Error: Unknown tool '{tool_name}'")
                print(f"Available tools: {', '.join(TOOLS_INFO.keys())}")
                return

            info = TOOLS_INFO[tool_name]
            args = custom_args if custom_args else info["args"]

            await call_tool(session, tool_name, args)


def main():
    """Main entry point."""
    try:
        if len(sys.argv) == 1:
            # Interactive mode
            asyncio.run(interactive_test())
        else:
            # CLI mode - test specific tool
            tool_name = sys.argv[1]

            # Parse additional arguments as JSON if provided
            custom_args = None
            if len(sys.argv) > 2:
                try:
                    custom_args = json.loads(sys.argv[2])
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON arguments: {sys.argv[2]}")
                    sys.exit(1)

            asyncio.run(test_specific_tool(tool_name, custom_args))
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
