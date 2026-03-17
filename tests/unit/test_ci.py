"""
Unit tests for CI environment detection.
"""
import os
import pytest
from unittest.mock import patch

from env_doctor.ci import detect_ci_environment, CIEnvironment


class TestCIDetection:
    """Test CI environment detection for each supported CI system."""

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_ENV": "/tmp/gh_env", "GITHUB_PATH": "/tmp/gh_path"}, clear=False)
    def test_github_actions(self):
        ci = detect_ci_environment()
        assert ci.name == "github_actions"
        assert ci.is_ci is True
        assert ci.env_persist_method == "github_env"
        assert ci.github_env_file == "/tmp/gh_env"
        assert ci.github_path_file == "/tmp/gh_path"

    @patch.dict(os.environ, {"GITLAB_CI": "true"}, clear=False)
    def test_gitlab_ci(self):
        ci = detect_ci_environment()
        assert ci.name == "gitlab_ci"
        assert ci.is_ci is True
        assert ci.env_persist_method == "gitlab_dotenv"

    @patch.dict(os.environ, {"CIRCLECI": "true"}, clear=False)
    def test_circleci(self):
        ci = detect_ci_environment()
        assert ci.name == "circleci"
        assert ci.is_ci is True
        assert ci.env_persist_method == "circleci_bash_env"

    @patch.dict(os.environ, {"TF_BUILD": "True"}, clear=False)
    def test_azure_pipelines(self):
        ci = detect_ci_environment()
        assert ci.name == "azure_pipelines"
        assert ci.is_ci is True
        assert ci.env_persist_method == "azure_vso"

    @patch.dict(os.environ, {"JENKINS_URL": "http://jenkins.local:8080"}, clear=False)
    def test_jenkins(self):
        ci = detect_ci_environment()
        assert ci.name == "jenkins"
        assert ci.is_ci is True
        assert ci.env_persist_method == "export_echo"

    @patch.dict(os.environ, {"CI": "true"}, clear=False)
    def test_generic_ci(self):
        ci = detect_ci_environment()
        assert ci.name == "generic_ci"
        assert ci.is_ci is True
        assert ci.env_persist_method == "export_echo"

    def test_local_environment(self):
        """Test local (non-CI) detection by clearing all CI env vars."""
        env_overrides = {
            "GITHUB_ACTIONS": "",
            "GITLAB_CI": "",
            "CIRCLECI": "",
            "TF_BUILD": "",
            "JENKINS_URL": "",
            "CI": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            ci = detect_ci_environment()
            assert ci.name == "local"
            assert ci.is_ci is False
            assert ci.env_persist_method in ("shell_rc", "windows_setx")

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITLAB_CI": "true"}, clear=False)
    def test_priority_github_over_gitlab(self):
        """GitHub Actions should take priority over GitLab CI."""
        ci = detect_ci_environment()
        assert ci.name == "github_actions"

    @patch.dict(os.environ, {"CI": "TRUE"}, clear=False)
    def test_ci_case_insensitive(self):
        """CI=TRUE (uppercase) should still be detected as generic CI."""
        # Our implementation checks .lower() == "true"
        ci = detect_ci_environment()
        assert ci.name == "generic_ci"
        assert ci.is_ci is True

    @patch('platform.system', return_value='Windows')
    def test_local_windows(self, mock_system):
        env_overrides = {
            "GITHUB_ACTIONS": "",
            "GITLAB_CI": "",
            "CIRCLECI": "",
            "TF_BUILD": "",
            "JENKINS_URL": "",
            "CI": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            ci = detect_ci_environment()
            assert ci.name == "local"
            assert ci.env_persist_method == "windows_setx"

    @patch('platform.system', return_value='Linux')
    def test_local_linux(self, mock_system):
        env_overrides = {
            "GITHUB_ACTIONS": "",
            "GITLAB_CI": "",
            "CIRCLECI": "",
            "TF_BUILD": "",
            "JENKINS_URL": "",
            "CI": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            ci = detect_ci_environment()
            assert ci.name == "local"
            assert ci.env_persist_method == "shell_rc"
