"""Unit tests for notebook HTML rendering and the public check() API."""
import importlib.util
from unittest.mock import patch

import pytest

_HAS_IPYTHON = importlib.util.find_spec("IPython") is not None

from env_doctor.report import format_result_html, is_notebook
from env_doctor.report.html import _badge
from env_doctor.api import CheckReport


@pytest.fixture
def sample_output():
    """A representative structured check-output dict (cli._build_check_output shape)."""
    return {
        "machine": {"hostname": "box", "platform": "Linux", "python_version": "3.11.4"},
        "status": "warning",
        "timestamp": "2026-06-15T00:00:00",
        "summary": {"driver": "found", "cuda": "found", "cudnn": "not_found", "issues_count": 2},
        "checks": {
            "wsl2": None,
            "driver": {
                "component": "nvidia_driver", "status": "success", "detected": True,
                "version": "535.1", "path": None,
                "metadata": {"detection_method": "pynvml", "gpu_count": 1},
                "issues": [], "recommendations": [],
            },
            "cuda": {
                "component": "cuda_toolkit", "status": "warning", "detected": True,
                "version": "12.1", "path": "/usr/local/cuda/bin/nvcc",
                "metadata": {"installation_count": 2},
                "issues": ["Multiple installs"], "recommendations": ["Set CUDA_HOME"],
            },
            "cudnn": None,
            "libraries": {
                "torch": {
                    "component": "python_library_torch", "status": "success",
                    "detected": True, "version": "2.1.0+cu121", "path": None,
                    "metadata": {"cuda_version": "12.1"}, "issues": [], "recommendations": [],
                },
                "tensorflow": {
                    "component": "tf", "status": "not_found", "detected": False,
                    "version": None, "path": None, "metadata": {},
                    "issues": [], "recommendations": [],
                },
            },
            "python_compat": {
                "component": "python_compat", "status": "success", "detected": True,
                "version": "3.11", "path": None,
                "metadata": {"constraints_checked": 5}, "issues": [], "recommendations": [],
            },
            "compute_compatibility": {
                "gpu_name": "RTX 4090", "arch_name": "Ada", "sm": "sm_89",
                "arch_list": ["sm_80", "sm_86"], "cuda_available": True,
                "status": "mismatch", "message": "PyTorch 2.1 does not support sm_89",
                "nightly_url": "https://download.pytorch.org/whl/nightly/cu121",
            },
        },
    }


class TestFormatResultHtml:
    def test_is_self_contained_fragment(self, sample_output):
        html = format_result_html(sample_output)
        assert html.startswith('<div class="env-doctor-report"')
        assert html.rstrip().endswith("</div>")
        # Self-contained: no external stylesheet / script tags.
        assert "<link" not in html
        assert "<script" not in html

    def test_renders_all_sections(self, sample_output):
        html = format_result_html(sample_output)
        for expected in ("NVIDIA Driver", "CUDA Toolkit", "torch", "tensorflow",
                         "Python Compatibility", "Compute Capability", "RTX 4090", "sm_89"):
            assert expected in html

    def test_issues_and_recommendations_render(self, sample_output):
        html = format_result_html(sample_output)
        assert "Multiple installs" in html
        assert "Set CUDA_HOME" in html

    def test_escapes_injected_values(self, sample_output):
        sample_output["machine"]["hostname"] = "<script>alert(1)</script>"
        html = format_result_html(sample_output)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_handles_missing_optional_checks(self):
        minimal = {
            "machine": {}, "status": "pass", "timestamp": "",
            "summary": {"issues_count": 0}, "checks": {},
        }
        html = format_result_html(minimal)
        assert html.startswith('<div class="env-doctor-report"')
        assert "No issues detected" in html

    @pytest.mark.parametrize("status,label", [
        ("success", "OK"), ("warning", "WARN"), ("error", "ERROR"),
        ("not_found", "ABSENT"), ("pass", "HEALTHY"), ("fail", "ISSUES FOUND"),
        ("mismatch", "MISMATCH"), ("anything-unknown", "UNKNOWN"),
    ])
    def test_badge_labels(self, status, label):
        assert label in _badge(status)


class TestIsNotebook:
    pytestmark = pytest.mark.skipif(not _HAS_IPYTHON, reason="IPython not installed")

    def test_false_when_no_kernel(self):
        # get_ipython() returns None in a terminal / no active kernel.
        with patch("IPython.get_ipython", return_value=None):
            assert is_notebook() is False

    def test_true_for_zmq_shell(self):
        class FakeShell:
            pass
        FakeShell.__name__ = "ZMQInteractiveShell"
        with patch("IPython.get_ipython", return_value=FakeShell()):
            assert is_notebook() is True

    def test_false_for_terminal_shell(self):
        class FakeShell:
            pass
        FakeShell.__name__ = "TerminalInteractiveShell"
        with patch("IPython.get_ipython", return_value=FakeShell()):
            assert is_notebook() is False


class TestCheckReport:
    def test_repr_html_matches_formatter(self, sample_output):
        report = CheckReport(sample_output)
        assert report._repr_html_() == format_result_html(sample_output)

    def test_repr_is_brief_summary(self, sample_output):
        report = CheckReport(sample_output)
        text = repr(report)
        assert "CheckReport" in text
        assert "status=warning" in text
        assert "issues=2" in text

    def test_to_dict_round_trips(self, sample_output):
        assert CheckReport(sample_output).to_dict() is sample_output

    def test_html_property(self, sample_output):
        assert CheckReport(sample_output).html.startswith('<div class="env-doctor-report"')


class TestCheckApi:
    def test_check_html_path_skips_text(self, sample_output):
        """format='html' returns a report and does not print the text diagnosis."""
        fake_bundle = {"output": sample_output}
        with patch("env_doctor.cli.collect_check_results", return_value=fake_bundle), \
             patch("env_doctor.cli.render_check_text") as mock_text:
            from env_doctor.api import check
            report = check(format="html")
        mock_text.assert_not_called()
        assert isinstance(report, CheckReport)
        assert report.to_dict() is sample_output

    def test_check_text_path_renders_text(self, sample_output):
        fake_bundle = {"output": sample_output}
        with patch("env_doctor.cli.collect_check_results", return_value=fake_bundle), \
             patch("env_doctor.cli.render_check_text") as mock_text:
            from env_doctor.api import check
            check(format="text")
        mock_text.assert_called_once_with(fake_bundle)