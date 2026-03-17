"""
CI environment detection for env-doctor.

Detects CI/CD systems and determines the appropriate method for
persisting environment variables (PATH, LD_LIBRARY_PATH, etc.).
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CIEnvironment:
    """Detected CI/CD environment information."""
    name: str  # "github_actions", "gitlab_ci", "jenkins", "circleci", "azure_pipelines", "generic_ci", "local"
    is_ci: bool
    env_persist_method: str  # "github_env", "gitlab_dotenv", "circleci_bash_env", "azure_vso", "export_echo", "shell_rc", "windows_setx"
    github_env_file: Optional[str] = None
    github_path_file: Optional[str] = None


def detect_ci_environment() -> CIEnvironment:
    """
    Detect the current CI/CD environment.

    Detection order:
    1. GitHub Actions (GITHUB_ACTIONS == "true")
    2. GitLab CI (GITLAB_CI == "true")
    3. CircleCI (CIRCLECI == "true")
    4. Azure Pipelines (TF_BUILD == "True")
    5. Jenkins (JENKINS_URL is set)
    6. Generic CI (CI == "true")
    7. Local (none of the above)

    Returns:
        CIEnvironment with detection results
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return CIEnvironment(
            name="github_actions",
            is_ci=True,
            env_persist_method="github_env",
            github_env_file=os.environ.get("GITHUB_ENV"),
            github_path_file=os.environ.get("GITHUB_PATH"),
        )

    if os.environ.get("GITLAB_CI") == "true":
        return CIEnvironment(
            name="gitlab_ci",
            is_ci=True,
            env_persist_method="gitlab_dotenv",
        )

    if os.environ.get("CIRCLECI") == "true":
        return CIEnvironment(
            name="circleci",
            is_ci=True,
            env_persist_method="circleci_bash_env",
        )

    if os.environ.get("TF_BUILD") == "True":
        return CIEnvironment(
            name="azure_pipelines",
            is_ci=True,
            env_persist_method="azure_vso",
        )

    if os.environ.get("JENKINS_URL"):
        return CIEnvironment(
            name="jenkins",
            is_ci=True,
            env_persist_method="export_echo",
        )

    if os.environ.get("CI", "").lower() == "true":
        return CIEnvironment(
            name="generic_ci",
            is_ci=True,
            env_persist_method="export_echo",
        )

    # Local environment
    import platform
    if platform.system() == "Windows":
        return CIEnvironment(
            name="local",
            is_ci=False,
            env_persist_method="windows_setx",
        )
    else:
        return CIEnvironment(
            name="local",
            is_ci=False,
            env_persist_method="shell_rc",
        )
