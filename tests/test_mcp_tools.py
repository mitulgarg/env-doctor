#!/usr/bin/env python3
"""
Test script for env-doctor MCP tools using JSON-RPC.

Tests all 10 MCP tools exposed by the env-doctor MCP server.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_all_tools():
    """Test all env-doctor MCP tools."""

    # Start the MCP server as a subprocess
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "env_doctor.mcp.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 80)
            print("TESTING ENV-DOCTOR MCP TOOLS")
            print("=" * 80)

            # List available tools
            print("\n[1/11] Listing available tools...")
            tools = await session.list_tools()
            print(f"✓ Found {len(tools.tools)} tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            # Test 1: env_check
            print("\n[2/11] Testing env_check...")
            try:
                result = await session.call_tool("env_check", {})
                data = json.loads(result.content[0].text)
                print(f"✓ env_check completed")
                print(f"  Status: {data['summary']['status']}")
                print(f"  Components checked: {data['summary']['component_count']}")
                print(f"  Components detected: {data['summary']['detected_count']}")
            except Exception as e:
                print(f"✗ env_check failed: {e}")

            # Test 2: env_check_component
            print("\n[3/11] Testing env_check_component (nvidia_driver)...")
            try:
                result = await session.call_tool("env_check_component", {"component": "nvidia_driver"})
                data = json.loads(result.content[0].text)
                print(f"✓ env_check_component completed")
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  Detected: {data.get('detected', False)}")
                if data.get('version'):
                    print(f"  Version: {data['version']}")
            except Exception as e:
                print(f"✗ env_check_component failed: {e}")

            # Test 3: cuda_info
            print("\n[4/11] Testing cuda_info...")
            try:
                result = await session.call_tool("cuda_info", {})
                data = json.loads(result.content[0].text)
                print(f"✓ cuda_info completed")
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  Detected: {data.get('detected', False)}")
                if data.get('version'):
                    print(f"  CUDA Version: {data['version']}")
                if data.get('metadata', {}).get('installation_count'):
                    print(f"  Installations: {data['metadata']['installation_count']}")
            except Exception as e:
                print(f"✗ cuda_info failed: {e}")

            # Test 4: cudnn_info
            print("\n[5/11] Testing cudnn_info...")
            try:
                result = await session.call_tool("cudnn_info", {})
                data = json.loads(result.content[0].text)
                print(f"✓ cudnn_info completed")
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  Detected: {data.get('detected', False)}")
                if data.get('version'):
                    print(f"  cuDNN Version: {data['version']}")
            except Exception as e:
                print(f"✗ cudnn_info failed: {e}")

            # Test 5: cuda_install (auto-detect)
            print("\n[6/11] Testing cuda_install (auto-detect)...")
            try:
                result = await session.call_tool("cuda_install", {})
                data = json.loads(result.content[0].text)
                print(f"✓ cuda_install completed")
                if 'error' in data:
                    print(f"  Message: {data['error']}")
                else:
                    print(f"  Platform: {data['platform']['os']} - {data['platform']['distro']} {data['platform']['distro_version']}")
                    print(f"  Recommended CUDA: {data.get('recommended_version', 'N/A')}")
                    if 'install_info' in data:
                        print(f"  Platform match: {data['install_info'].get('label', 'Unknown')}")
                        print(f"  Steps: {len(data['install_info'].get('steps', []))} installation steps")
            except Exception as e:
                print(f"✗ cuda_install failed: {e}")

            # Test 6: cuda_install (specific version)
            print("\n[7/11] Testing cuda_install (version=12.4)...")
            try:
                result = await session.call_tool("cuda_install", {"version": "12.4"})
                data = json.loads(result.content[0].text)
                print(f"✓ cuda_install (12.4) completed")
                if 'error' in data:
                    print(f"  Message: {data['error']}")
                else:
                    print(f"  Requested version: {data.get('requested_version', 'N/A')}")
                    if 'install_info' in data:
                        print(f"  Platform match: {data['install_info'].get('label', 'Unknown')}")
            except Exception as e:
                print(f"✗ cuda_install (12.4) failed: {e}")

            # Test 7: install_command
            print("\n[8/11] Testing install_command (torch)...")
            try:
                result = await session.call_tool("install_command", {"library": "torch"})
                data = json.loads(result.content[0].text)
                print(f"✓ install_command completed")
                print(f"  Library: {data['library']}")
                print(f"  Driver detected: {data.get('driver_detected', False)}")
                if data.get('driver_detected'):
                    print(f"  Max CUDA: {data.get('max_cuda', 'N/A')}")
                print(f"  Command: {data.get('install_command', 'N/A')[:80]}...")
            except Exception as e:
                print(f"✗ install_command failed: {e}")

            # Test 8: model_list
            print("\n[9/11] Testing model_list...")
            try:
                result = await session.call_tool("model_list", {})
                data = json.loads(result.content[0].text)
                print(f"✓ model_list completed")
                if 'models_by_category' in data:
                    for category, models in data['models_by_category'].items():
                        print(f"  {category}: {len(models)} models")
                if 'stats' in data:
                    print(f"  Total models: {data['stats'].get('total_models', 0)}")
            except Exception as e:
                print(f"✗ model_list failed: {e}")

            # Test 9: model_check
            print("\n[10/11] Testing model_check (llama-3-8b)...")
            try:
                result = await session.call_tool("model_check", {"model_name": "llama-3-8b"})
                data = json.loads(result.content[0].text)
                print(f"✓ model_check completed")
                print(f"  Success: {data.get('success', False)}")
                if data.get('success'):
                    print(f"  Model: {data.get('model_name', 'N/A')}")
                    print(f"  GPU available: {data.get('gpu_info', {}).get('available', False)}")
                    if data.get('compatibility'):
                        fits = any(v.get('fits', False) for v in data['compatibility'].get('fits_on_single_gpu', {}).values())
                        print(f"  Fits on GPU: {fits}")
            except Exception as e:
                print(f"✗ model_check failed: {e}")

            # Test 10: dockerfile_validate
            print("\n[11/11] Testing dockerfile_validate...")
            dockerfile_content = """FROM python:3.10
RUN pip install torch torchvision torchaudio
CMD ["python", "app.py"]
"""
            try:
                result = await session.call_tool("dockerfile_validate", {"content": dockerfile_content})
                data = json.loads(result.content[0].text)
                print(f"✓ dockerfile_validate completed")
                print(f"  Success: {data.get('success', False)}")
                print(f"  Errors: {data.get('error_count', 0)}")
                print(f"  Warnings: {data.get('warning_count', 0)}")
                print(f"  Issues found: {len(data.get('issues', []))}")
            except Exception as e:
                print(f"✗ dockerfile_validate failed: {e}")

            # Test 11: docker_compose_validate
            print("\n[12/11] Testing docker_compose_validate...")
            compose_content = """version: '3.8'
services:
  app:
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: python train.py
"""
            try:
                result = await session.call_tool("docker_compose_validate", {"content": compose_content})
                data = json.loads(result.content[0].text)
                print(f"✓ docker_compose_validate completed")
                print(f"  Success: {data.get('success', False)}")
                print(f"  Errors: {data.get('error_count', 0)}")
                print(f"  Warnings: {data.get('warning_count', 0)}")
                print(f"  Issues found: {len(data.get('issues', []))}")
            except Exception as e:
                print(f"✗ docker_compose_validate failed: {e}")

            print("\n" + "=" * 80)
            print("TESTING COMPLETE")
            print("=" * 80)


def main():
    """Main entry point."""
    try:
        asyncio.run(test_all_tools())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
