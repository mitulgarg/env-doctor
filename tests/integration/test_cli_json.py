"""
Integration tests for JSON output in CLI commands.

Tests verify that --json and --ci flags work correctly and produce
valid, parseable JSON output with proper exit codes.
"""
import json
import subprocess
import sys
import pytest


def test_check_json_output_valid():
    """Test that check command with --json produces valid JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check", "--json"],
        capture_output=True,
        text=True
    )

    # Should produce valid JSON output
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {e}\nOutput: {result.stdout}")

    # Verify required top-level keys
    assert "status" in data, "Missing 'status' key in JSON output"
    assert "timestamp" in data, "Missing 'timestamp' key in JSON output"
    assert "summary" in data, "Missing 'summary' key in JSON output"
    assert "checks" in data, "Missing 'checks' key in JSON output"

    # Verify status is one of expected values
    assert data["status"] in ["pass", "warning", "fail"], \
        f"Invalid status value: {data['status']}"

    # Verify summary structure
    assert "driver" in data["summary"]
    assert "cuda" in data["summary"]
    assert "issues_count" in data["summary"]

    # Verify checks structure
    assert "driver" in data["checks"]
    assert "cuda" in data["checks"]
    assert "libraries" in data["checks"]


def test_check_ci_flag_implies_json():
    """Test that --ci flag produces JSON output."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check", "--ci"],
        capture_output=True,
        text=True
    )

    # Should produce valid JSON output
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"--ci flag should produce JSON output: {e}")

    assert "status" in data
    assert "checks" in data


def test_check_json_exit_codes():
    """Test that check command exits with proper codes."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check", "--json"],
        capture_output=True,
        text=True
    )

    # Exit code should be 0, 1, or 2
    assert result.returncode in [0, 1, 2], \
        f"Unexpected exit code: {result.returncode}"

    # Parse JSON to verify exit code matches status
    data = json.loads(result.stdout)

    # Exit code should match status
    if data["status"] == "pass":
        # Could be 0 or 1 depending on warnings
        assert result.returncode in [0, 1]
    elif data["status"] == "warning":
        assert result.returncode == 1
    elif data["status"] == "fail":
        assert result.returncode in [1, 2]


def test_cuda_info_json_output():
    """Test cuda-info command with --json flag."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "cuda-info", "--json"],
        capture_output=True,
        text=True
    )

    # Should produce valid JSON
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {e}\nOutput: {result.stdout}")

    # Verify structure matches DetectionResult
    assert "component" in data
    assert "status" in data
    assert "detected" in data
    assert isinstance(data["detected"], bool)
    assert "metadata" in data
    assert "issues" in data
    assert "recommendations" in data


def test_cudnn_info_json_output():
    """Test cudnn-info command with --json flag."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "cudnn-info", "--json"],
        capture_output=True,
        text=True
    )

    # Should produce valid JSON (or error JSON if platform not supported)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {e}\nOutput: {result.stdout}")

    # Either a DetectionResult or an error
    if "error" in data:
        assert isinstance(data["error"], str)
    else:
        assert "component" in data
        assert "status" in data
        assert "detected" in data


def test_scan_json_output():
    """Test scan command with --json flag."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "scan", "--json"],
        capture_output=True,
        text=True,
        cwd="/tmp"  # Use temp dir to avoid scanning actual code
    )

    # Should produce valid JSON
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {e}\nOutput: {result.stdout}")

    # Verify structure
    assert "status" in data
    assert "timestamp" in data
    assert "dependencies" in data
    assert "issues" in data
    assert "recommendations" in data
    assert isinstance(data["dependencies"], list)


def test_check_json_components_structure():
    """Test that check JSON includes all expected components."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check", "--json"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)
    checks = data["checks"]

    # Verify each component has proper structure
    for component_name, component_data in checks.items():
        if component_data is None:
            continue

        if isinstance(component_data, dict) and "component" in component_data:
            # Single component
            assert "status" in component_data
            assert "detected" in component_data
            assert isinstance(component_data["detected"], bool)
            assert "metadata" in component_data
            assert "issues" in component_data
            assert "recommendations" in component_data
        elif isinstance(component_data, dict):
            # Libraries dict
            for lib_name, lib_data in component_data.items():
                assert "component" in lib_data
                assert "status" in lib_data
                assert "detected" in lib_data


def test_json_output_no_emoji():
    """Test that JSON output contains no emoji characters."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check", "--json"],
        capture_output=True,
        text=True
    )

    # Common emoji characters used in human output
    emojis = ["🩺", "✅", "❌", "⚠️", "📦", "→", "ℹ️", "🔧", "🧠"]

    for emoji in emojis:
        assert emoji not in result.stdout, \
            f"JSON output should not contain emoji: {emoji}"


def test_default_output_not_json():
    """Test that default output (without --json) is NOT JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check"],
        capture_output=True,
        text=True
    )

    # Default output should contain emojis (not JSON)
    # Just check that it's not valid JSON
    try:
        json.loads(result.stdout)
        pytest.fail("Default output should not be valid JSON")
    except json.JSONDecodeError:
        # Expected - default output is human-readable, not JSON
        pass


def test_json_timestamp_format():
    """Test that timestamp in JSON is in ISO 8601 format."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check", "--json"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    # Verify timestamp exists and is ISO 8601 format
    assert "timestamp" in data
    timestamp = data["timestamp"]

    # Try to parse as ISO format
    from datetime import datetime
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as e:
        pytest.fail(f"Timestamp is not in ISO 8601 format: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
