"""Tests for the export and validate CLI commands."""
import json
import platform
from unittest.mock import patch, MagicMock

import pytest

from env_doctor.cli import (
    _collect_env_spec,
    _compare_versions,
    _validate_spec,
    export_command,
    validate_command,
)
from env_doctor.core.detector import DetectionResult, Status


# ── Fixtures ──────────────────────────────────────────────────

def _make_driver_result(version="555.42.02", detected=True):
    return DetectionResult(
        component="nvidia_driver",
        status=Status.SUCCESS if detected else Status.NOT_FOUND,
        version=version if detected else None,
        metadata={
            "max_cuda_version": "12.4",
            "gpu_count": 1,
            "gpus": [
                {"name": "RTX 4090", "total_vram_mb": 24564, "compute_capability": "8.9"}
            ],
            "primary_gpu_name": "RTX 4090",
        } if detected else {},
    )


def _make_cuda_result(version="12.4", detected=True):
    return DetectionResult(
        component="cuda_toolkit",
        status=Status.SUCCESS if detected else Status.NOT_FOUND,
        version=version if detected else None,
    )


def _make_cudnn_result(version="8.9.7", detected=True):
    return DetectionResult(
        component="cudnn",
        status=Status.SUCCESS if detected else Status.NOT_FOUND,
        version=version if detected else None,
    )


def _make_lib_result(lib_name, version="2.3.0", cuda_version="12.1", detected=True):
    return DetectionResult(
        component=f"python_library_{lib_name}",
        status=Status.SUCCESS if detected else Status.NOT_FOUND,
        version=version if detected else None,
        metadata={"cuda_version": cuda_version} if detected else {},
    )


def _make_python_compat_result():
    return DetectionResult(
        component="python_compat",
        status=Status.SUCCESS,
        version=platform.python_version()[:4],  # e.g. "3.12"
        metadata={"python_full_version": platform.python_version()},
    )


def _mock_detector(result, can_run=True):
    det = MagicMock()
    det.detect.return_value = result
    det.can_run.return_value = can_run
    return det


# ── _compare_versions ────────────────────────────────────────

class TestCompareVersions:
    def test_exact_match(self):
        assert _compare_versions("12.4", "12.4", strict=True) == "match"
        assert _compare_versions("12.4", "12.4", strict=False) == "match"

    def test_none_expected(self):
        assert _compare_versions("12.4", None, strict=True) == "match"

    def test_none_actual(self):
        assert _compare_versions(None, "12.4", strict=True) == "mismatch"

    def test_strict_different(self):
        assert _compare_versions("12.4.1", "12.4.0", strict=True) == "mismatch"

    def test_compatible_minor(self):
        assert _compare_versions("12.4.1", "12.4.0", strict=False) == "compatible"

    def test_incompatible_minor(self):
        assert _compare_versions("12.3.0", "12.4.0", strict=False) == "mismatch"


# ── _collect_env_spec ────────────────────────────────────────

class TestCollectEnvSpec:
    @patch("env_doctor.cli.DetectorRegistry")
    @patch("env_doctor.cli.PythonLibraryDetector")
    def test_basic_spec_structure(self, mock_lib_cls, mock_registry):
        # Setup mocks
        mock_registry.get.side_effect = lambda name: {
            "wsl2": _mock_detector(None, can_run=False),
            "nvidia_driver": _mock_detector(_make_driver_result()),
            "cuda_toolkit": _mock_detector(_make_cuda_result()),
            "cudnn": _mock_detector(_make_cudnn_result()),
            "python_compat": _mock_detector(_make_python_compat_result()),
        }[name]

        mock_lib_cls.side_effect = lambda lib: _mock_detector(
            _make_lib_result(lib, detected=(lib == "torch"))
        )

        spec = _collect_env_spec()

        assert "env_doctor_version" in spec
        assert spec["spec_version"] == "1.0"
        assert "exported_at" in spec
        assert "platform" in spec
        assert "environment" in spec

        env = spec["environment"]
        assert env["nvidia_driver"] == "555.42.02"
        assert env["cuda_toolkit"] == "12.4"
        assert env["cudnn"] == "8.9.7"
        assert env["python"] is not None
        assert env["libraries"]["torch"]["version"] == "2.3.0"
        assert env["libraries"]["tensorflow"] is None
        assert len(env["gpus"]) == 1
        assert env["gpus"][0]["name"] == "RTX 4090"


# ── _validate_spec ───────────────────────────────────────────

class TestValidateSpec:
    def _make_spec(self, **overrides):
        base = {
            "env_doctor_version": "0.2.6",
            "spec_version": "1.0",
            "exported_at": "2026-03-12T00:00:00",
            "platform": {"os": "linux"},
            "environment": {
                "nvidia_driver": "555.42.02",
                "max_cuda_version": "12.4",
                "cuda_toolkit": "12.4",
                "cudnn": "8.9.7",
                "python": platform.python_version(),
                "libraries": {
                    "torch": {"version": "2.3.0", "cuda_version": "12.1"},
                    "tensorflow": None,
                    "jax": None,
                },
                "gpus": [{"name": "RTX 4090", "vram_mb": 24564, "compute_capability": "8.9"}],
            },
        }
        base["environment"].update(overrides)
        return base

    @patch("env_doctor.cli._collect_env_spec")
    def test_identical_environment_passes(self, mock_collect):
        spec = self._make_spec()
        mock_collect.return_value = spec.copy()

        report = _validate_spec(spec, strict=False)
        assert report["status"] == "pass"
        assert all(c["status"] == "match" for c in report["comparisons"])

    @patch("env_doctor.cli._collect_env_spec")
    def test_mismatch_detected(self, mock_collect):
        spec = self._make_spec()
        current = self._make_spec(cuda_toolkit="11.8")
        mock_collect.return_value = current

        report = _validate_spec(spec, strict=True)
        assert report["status"] == "fail"

        cuda_comp = next(c for c in report["comparisons"] if c["component"] == "CUDA Toolkit")
        assert cuda_comp["status"] == "mismatch"

    @patch("env_doctor.cli._collect_env_spec")
    def test_compatible_mode(self, mock_collect):
        spec = self._make_spec(cuda_toolkit="12.4.0")
        current = self._make_spec(cuda_toolkit="12.4.1")
        mock_collect.return_value = current

        report = _validate_spec(spec, strict=False)
        # 12.4.1 vs 12.4.0 should be compatible
        cuda_comp = next(c for c in report["comparisons"] if c["component"] == "CUDA Toolkit")
        assert cuda_comp["status"] == "compatible"

    @patch("env_doctor.cli._collect_env_spec")
    def test_missing_library(self, mock_collect):
        spec = self._make_spec()
        spec["environment"]["libraries"]["torch"] = {"version": "2.3.0", "cuda_version": "12.1"}
        current = self._make_spec()
        current["environment"]["libraries"]["torch"] = None
        mock_collect.return_value = current

        report = _validate_spec(spec, strict=False)
        torch_comp = next(c for c in report["comparisons"] if "torch" in c["component"] and "CUDA" not in c["component"])
        assert torch_comp["status"] == "missing"


# ── export_command ───────────────────────────────────────────

class TestExportCommand:
    @patch("env_doctor.cli._collect_env_spec")
    def test_export_to_stdout(self, mock_collect, capsys):
        mock_collect.return_value = {"spec_version": "1.0", "test": True}
        export_command(output_file=None)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["spec_version"] == "1.0"

    @patch("env_doctor.cli._collect_env_spec")
    def test_export_to_file(self, mock_collect, tmp_path):
        mock_collect.return_value = {"spec_version": "1.0", "test": True}
        out_file = str(tmp_path / "spec.json")
        export_command(output_file=out_file)
        with open(out_file) as f:
            data = json.load(f)
        assert data["spec_version"] == "1.0"


# ── validate_command ─────────────────────────────────────────

class TestValidateCommand:
    @patch("env_doctor.cli._validate_spec")
    def test_validate_json_output(self, mock_validate, tmp_path, capsys):
        spec = {"environment": {"nvidia_driver": "555.42.02"}}
        spec_file = str(tmp_path / "spec.json")
        with open(spec_file, "w") as f:
            json.dump(spec, f)

        mock_validate.return_value = {
            "status": "pass",
            "timestamp": "2026-03-12T00:00:00",
            "spec_source": "test",
            "mode": "compatible",
            "comparisons": [],
        }

        with pytest.raises(SystemExit) as exc_info:
            validate_command(spec_file, strict=False, output_json=True)
        assert exc_info.value.code == 0

    def test_validate_missing_file(self):
        with pytest.raises(SystemExit) as exc_info:
            validate_command("/nonexistent/file.json")
        assert exc_info.value.code == 1

    def test_validate_invalid_json(self, tmp_path):
        bad_file = str(tmp_path / "bad.json")
        with open(bad_file, "w") as f:
            f.write("not json")
        with pytest.raises(SystemExit) as exc_info:
            validate_command(bad_file)
        assert exc_info.value.code == 1

    def test_validate_missing_environment_key(self, tmp_path):
        spec_file = str(tmp_path / "no_env.json")
        with open(spec_file, "w") as f:
            json.dump({"foo": "bar"}, f)
        with pytest.raises(SystemExit) as exc_info:
            validate_command(spec_file)
        assert exc_info.value.code == 1
