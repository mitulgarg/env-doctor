"""
Integration tests for inference-engine resolution in the CLI
(`env-doctor install vllm`, `env-doctor check --engine ...`).

The L40S / CUDA 12.6 case from issue #139 is the canonical fixture.
"""
import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from env_doctor import cli, db
from env_doctor.core.detector import DetectionResult, Status

BUNDLED = os.path.join(os.path.dirname(db.__file__), "data", "inference_engines.json")


@pytest.fixture
def engine_data():
    with open(BUNDLED, "r") as f:
        data = json.load(f)
    with patch("env_doctor.db.load_engine_data", return_value=data):
        yield data


def _driver(max_cuda="12.6", version="560.35.03"):
    result = MagicMock()
    result.detected = True
    result.version = version
    result.metadata = {"max_cuda_version": max_cuda}
    detector = MagicMock()
    detector.detect.return_value = result
    return detector


def _not_installed(name):
    return DetectionResult(component=f"python_library_{name}", status=Status.NOT_FOUND)


class TestEngineInstall:
    @patch("shutil.which", return_value="/usr/bin/uv")
    @patch("env_doctor.core.registry.DetectorRegistry.get")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_install_vllm_on_cuda126_box(self, mock_stdout, mock_registry, _which, engine_data):
        mock_registry.return_value = _driver("12.6")

        cli.install_command("vllm")
        out = mock_stdout.getvalue()

        assert "PRESCRIPTION FOR: vllm" in out
        assert "Default wheel: CUDA 13.0" in out
        assert "Copy-to-fix: uv pip install vllm==" in out
        assert "--torch-backend=cu129" in out
        assert "RISKY" in out  # driver upgrade offered last, flagged

    @patch("shutil.which", return_value="/usr/bin/uv")
    @patch("env_doctor.core.registry.DetectorRegistry.get")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_install_sglang_pinned_version(self, mock_stdout, mock_registry, _which, engine_data):
        mock_registry.return_value = _driver("12.6")

        cli.install_command("sglang@0.5.19")
        out = mock_stdout.getvalue()

        assert "SGLang 0.5.19" in out
        assert "https://docs.sglang.ai/whl/cu129/" in out
        assert "sglang-kernel==0.4.6.post1" in out

    @patch("env_doctor.cli._run_install_command", return_value=True)
    @patch("shutil.which", return_value="/usr/bin/uv")
    @patch("env_doctor.core.registry.DetectorRegistry.get")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_execute_runs_safe_option_commands_in_order(self, _out, mock_registry, _which, mock_run, engine_data):
        mock_registry.return_value = _driver("12.6")

        cli.install_command("sglang@0.5.19", execute=True)

        cmds = [c.args[0] for c in mock_run.call_args_list]
        assert len(cmds) == 4
        assert cmds[0] == "uv pip install sglang==0.5.19"

    @patch("env_doctor.cli._run_install_command")
    @patch("shutil.which", return_value="/usr/bin/uv")
    @patch("env_doctor.core.registry.DetectorRegistry.get")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_execute_refuses_when_only_driver_upgrade(self, mock_stdout, mock_registry, _which, mock_run, engine_data):
        mock_registry.return_value = _driver("11.8", "520.00")

        cli.install_command("vllm", execute=True)

        mock_run.assert_not_called()
        assert "not executing" in mock_stdout.getvalue()


class TestCheckEngines:
    @patch("shutil.which", return_value="/usr/bin/uv")
    @patch("env_doctor.detectors.python_libraries.PythonLibraryDetector.detect")
    def test_requested_engine_resolved_and_serialized(self, mock_detect, _which, engine_data):
        mock_detect.side_effect = lambda: _not_installed("x")

        resolutions = cli.collect_engine_results(["vllm"], "12.6")
        assert list(resolutions) == ["vllm"]

        det = resolutions["vllm"].to_detection_result()
        assert det.status == Status.WARNING
        d = det.to_dict()
        assert d["metadata"]["copy_to_fix"].startswith("uv pip install vllm==")
        json.dumps(d)  # JSON-serializable for --json / --report-to

    @patch("env_doctor.detectors.python_libraries.PythonLibraryDetector.detect")
    def test_nothing_requested_or_installed_skips_db(self, mock_detect):
        mock_detect.side_effect = lambda: _not_installed("x")
        with patch("env_doctor.db.load_engine_data") as load:
            assert cli.collect_engine_results(None, "12.6") == {}
            load.assert_not_called()

    @patch("shutil.which", return_value="/usr/bin/uv")
    def test_installed_engine_auto_checked_with_torch_cuda(self, _which, engine_data):
        installed = DetectionResult(
            component="python_library_vllm", status=Status.SUCCESS, version="0.30.0",
            metadata={"engine_version": "0.30.0", "engine_cuda": None,
                      "torch_version": "2.13.0", "torch_cuda": None, "kernel_libs": {}},
        )
        torch_result = DetectionResult(component="python_library_torch", status=Status.SUCCESS,
                                       version="2.13.0", metadata={"cuda_version": "13.0"})

        def detect_for(name):
            return installed if name == "vllm" else _not_installed(name)

        with patch("env_doctor.detectors.python_libraries.PythonLibraryDetector.__init__",
                   lambda self, n=None: setattr(self, "library_name", n)), \
             patch("env_doctor.detectors.python_libraries.PythonLibraryDetector.detect",
                   lambda self: detect_for(self.library_name)):
            resolutions = cli.collect_engine_results(None, "12.6", torch_result=torch_result)

        res = resolutions["vllm"]
        assert res.installed_cuda == "13.0"  # filled in from torch.version.cuda
        assert res.status == "error"
        assert "cu129" in res.copy_to_fix

    def test_engines_in_check_output_and_status(self, engine_data):
        from env_doctor.engines import resolve_engine
        res = resolve_engine("vllm", "11.8", engine_data)
        results = {"engines": {"vllm": res.to_detection_result()}}

        assert cli.determine_overall_status(results) == "fail"
        assert cli.determine_exit_code(results) == 2
        assert cli.count_issues(results) >= 1

    def test_html_report_renders_engine_section(self, engine_data):
        from env_doctor.engines import resolve_engine
        from env_doctor.report.html import format_result_html

        det = resolve_engine("sglang", "12.6", engine_data).to_detection_result()
        html = format_result_html({
            "status": "warning", "summary": {"issues_count": 1},
            "checks": {"engines": {"sglang": det.to_dict()}},
        })
        assert "Inference Engine: sglang" in html
