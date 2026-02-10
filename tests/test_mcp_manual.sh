#!/bin/bash
# Manual JSON-RPC testing script for env-doctor MCP server
#
# This script demonstrates how to manually test MCP tools using JSON-RPC
# over stdio. Each test sends a JSON-RPC request and captures the response.

echo "========================================================================"
echo "ENV-DOCTOR MCP MANUAL TESTING"
echo "========================================================================"

# Helper function to send JSON-RPC request
send_request() {
    local method=$1
    local params=$2
    local id=$3

    echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"$method\",\"params\":$params}"
}

# Start the MCP server
MCP_SERVER="python -m env_doctor.mcp.server"

echo ""
echo "[Test 1] Initialize connection"
echo "----------------------------------------"
send_request "initialize" '{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test-client","version":"1.0"}}' 1 | $MCP_SERVER &
SERVER_PID=$!
sleep 1

echo ""
echo "[Test 2] List tools"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | $MCP_SERVER

echo ""
echo "[Test 3] Call env_check"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"env_check","arguments":{}}}' | $MCP_SERVER

echo ""
echo "[Test 4] Call cuda_info"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"cuda_info","arguments":{}}}' | $MCP_SERVER

echo ""
echo "[Test 5] Call install_command for torch"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"install_command","arguments":{"library":"torch"}}}' | $MCP_SERVER

echo ""
echo "[Test 6] Call cuda_install (auto-detect)"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"cuda_install","arguments":{}}}' | $MCP_SERVER

echo ""
echo "[Test 7] Call cuda_install (version 12.4)"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"cuda_install","arguments":{"version":"12.4"}}}' | $MCP_SERVER

echo ""
echo "[Test 8] Call model_check"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"model_check","arguments":{"model_name":"llama-3-8b"}}}' | $MCP_SERVER

echo ""
echo "[Test 9] Call model_list"
echo "----------------------------------------"
echo '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"model_list","arguments":{}}}' | $MCP_SERVER

echo ""
echo "[Test 10] Call dockerfile_validate"
echo "----------------------------------------"
DOCKERFILE_CONTENT='FROM python:3.10\nRUN pip install torch\nCMD ["python", "app.py"]'
echo "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"tools/call\",\"params\":{\"name\":\"dockerfile_validate\",\"arguments\":{\"content\":\"$DOCKERFILE_CONTENT\"}}}" | $MCP_SERVER

echo ""
echo "========================================================================"
echo "MANUAL TESTING COMPLETE"
echo "========================================================================"

# Cleanup
kill $SERVER_PID 2>/dev/null
