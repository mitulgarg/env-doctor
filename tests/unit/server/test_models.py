"""Tests for SQLAlchemy models and Pydantic validation."""
import json

import pytest

# Skip all tests if dashboard dependencies aren't installed
pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from env_doctor.server.routes import ReportPayload, MachineInfo, _extract_fields


SAMPLE_REPORT = {
    "machine": {
        "machine_id": "abc-123",
        "hostname": "gpu-node-01",
        "platform": "Linux",
        "platform_release": "5.15.0",
        "python_version": "3.11.5",
        "reported_at": "2025-01-01T00:00:00+00:00",
    },
    "status": "pass",
    "timestamp": "2025-01-01T00:00:00",
    "summary": {
        "driver": "found",
        "cuda": "found",
        "cudnn": "not_found",
        "issues_count": 1,
    },
    "checks": {
        "wsl2": None,
        "driver": {
            "component": "nvidia_driver",
            "status": "success",
            "detected": True,
            "version": "535.129.03",
            "path": None,
            "metadata": {"primary_gpu_name": "NVIDIA A100-SXM4-80GB"},
            "issues": [],
            "recommendations": [],
        },
        "cuda": {
            "component": "cuda_toolkit",
            "status": "success",
            "detected": True,
            "version": "12.2",
            "path": "/usr/local/cuda",
            "metadata": {},
            "issues": [],
            "recommendations": [],
        },
        "cudnn": None,
        "libraries": {
            "torch": {
                "component": "torch",
                "status": "success",
                "detected": True,
                "version": "2.1.0",
                "path": None,
                "metadata": {"cuda_version": "12.1"},
                "issues": [],
                "recommendations": [],
            },
        },
        "python_compat": {
            "component": "python_compat",
            "status": "success",
            "detected": True,
            "version": "3.11.5",
            "path": None,
            "metadata": {},
            "issues": [],
            "recommendations": [],
        },
        "compute_compatibility": None,
    },
}


class TestReportPayload:
    def test_parses_valid_report(self):
        payload = ReportPayload(**SAMPLE_REPORT)
        assert payload.status == "pass"
        assert payload.machine.machine_id == "abc-123"
        assert payload.machine.hostname == "gpu-node-01"

    def test_extracts_fields(self):
        payload = ReportPayload(**SAMPLE_REPORT)
        fields = _extract_fields(payload)
        assert fields["gpu_name"] == "NVIDIA A100-SXM4-80GB"
        assert fields["driver_version"] == "535.129.03"
        assert fields["cuda_version"] == "12.2"
        assert fields["torch_version"] == "2.1.0"
        assert fields["summary_issues_count"] == 1

    def test_handles_missing_optional_fields(self):
        minimal = {
            "machine": {"machine_id": "x", "hostname": "h"},
            "status": "fail",
            "timestamp": "2025-01-01T00:00:00",
            "summary": {"driver": "not_found", "cuda": "not_found", "issues_count": 0},
            "checks": {},
        }
        payload = ReportPayload(**minimal)
        fields = _extract_fields(payload)
        assert fields["gpu_name"] is None
        assert fields["driver_version"] is None

    def test_serializes_to_json(self):
        payload = ReportPayload(**SAMPLE_REPORT)
        data = json.dumps(payload.model_dump(), default=str)
        assert "abc-123" in data
