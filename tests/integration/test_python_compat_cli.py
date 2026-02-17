"""
Integration tests for python-compat CLI command.
"""
import json
import subprocess
import sys

import pytest


def test_python_compat_command_produces_output():
    """Test that python-compat command runs and produces output."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "python-compat"],
        capture_output=True,
        timeout=30,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    assert "PYTHON VERSION COMPATIBILITY CHECK" in stdout
    assert "Python Version:" in stdout


def test_python_compat_json_output():
    """Test that python-compat --json produces valid JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "python-compat", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {e}\nOutput: {result.stdout}")

    assert data["component"] == "python_compat"
    assert "status" in data
    assert "version" in data
    assert "metadata" in data
    assert "conflicts" in data["metadata"]
    assert "cascades" in data["metadata"]


def test_check_command_includes_python_compat():
    """Test that check --json includes python_compat in output."""
    result = subprocess.run(
        [sys.executable, "-m", "env_doctor.cli", "check", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {e}\nOutput: {result.stdout}")

    assert "python_compat" in data["checks"], "python_compat missing from check output"
    assert data["checks"]["python_compat"]["component"] == "python_compat"
