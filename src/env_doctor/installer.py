"""
CUDA Toolkit installation execution engine.

Executes the installation steps from cuda_toolkit_install.json,
with real-time output streaming, logging, and CI-aware env persistence.
"""
import os
import sys
import subprocess
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .ci import CIEnvironment, detect_ci_environment
from .core.exceptions import InstallerError, PrivilegeError, StepExecutionError


@dataclass
class StepResult:
    """Result of executing a single installation step."""
    command: str
    phase: str  # "install", "post_install", "verify"
    success: bool
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass
class InstallResult:
    """Result of the full installation process."""
    success: bool
    cuda_version: str
    platform_key: str
    steps_completed: List[StepResult] = field(default_factory=list)
    steps_remaining: List[str] = field(default_factory=list)
    env_vars_set: Dict[str, str] = field(default_factory=dict)
    verification_passed: bool = False
    error_message: Optional[str] = None
    log_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "cuda_version": self.cuda_version,
            "platform_key": self.platform_key,
            "steps_completed": [
                {
                    "command": s.command,
                    "phase": s.phase,
                    "success": s.success,
                    "return_code": s.return_code,
                    "duration_seconds": s.duration_seconds,
                }
                for s in self.steps_completed
            ],
            "steps_remaining": self.steps_remaining,
            "env_vars_set": self.env_vars_set,
            "verification_passed": self.verification_passed,
            "error_message": self.error_message,
            "log_file": self.log_file,
        }


class CudaInstaller:
    """
    Executes CUDA Toolkit installation steps.

    Reads steps from install_info (from cuda_toolkit_install.json),
    runs them sequentially, handles env persistence, and verifies.
    """

    def __init__(self, install_info, cuda_version, platform_info, ci_env=None,
                 dry_run=False, yes=False):
        """
        Args:
            install_info: Dict from get_cuda_install_steps() with steps, post_install, verify, etc.
            cuda_version: CUDA version string (e.g., "12.6")
            platform_info: Dict from detect_platform()
            ci_env: CIEnvironment instance (auto-detected if None)
            dry_run: If True, show commands without executing
            yes: If True, skip confirmation prompts
        """
        self.install_info = install_info
        self.cuda_version = cuda_version
        self.platform_info = platform_info
        self.ci_env = ci_env or detect_ci_environment()
        self.dry_run = dry_run
        self.yes = yes
        self.log_path = None
        self.log_file_handle = None

    def run(self) -> InstallResult:
        """
        Execute the full installation process.

        Returns:
            InstallResult with details of what happened
        """
        result = InstallResult(
            success=False,
            cuda_version=self.cuda_version,
            platform_key=self.platform_info.get("platform_key", "unknown"),
        )

        try:
            self._init_log()
            result.log_file = str(self.log_path) if self.log_path else None

            steps = self.install_info.get("steps", [])
            if not steps:
                result.error_message = "No installation steps found"
                return result

            # Check privileges (skip for dry run)
            if not self.dry_run:
                self._check_privileges(steps)

            # Prompt for confirmation
            if not self.dry_run and not self.yes:
                if not self._prompt_confirmation(steps):
                    result.error_message = "Installation cancelled by user"
                    return result

            # Execute installation steps
            all_steps = list(steps)
            for i, step in enumerate(steps):
                step_result = self._execute_step(step, "install", i + 1, len(steps))
                result.steps_completed.append(step_result)

                if not step_result.success:
                    result.steps_remaining = all_steps[i + 1:]
                    result.error_message = (
                        f"Step {i + 1} failed (exit code {step_result.return_code}): {step}"
                    )
                    self._log(f"FAILED: {result.error_message}")
                    return result

            # Persist environment variables
            env_vars = self._persist_env_vars()
            result.env_vars_set = env_vars

            # Run verification
            verify_cmd = self.install_info.get("verify")
            if verify_cmd:
                verify_result = self._execute_step(verify_cmd, "verify", 1, 1)
                result.steps_completed.append(verify_result)
                result.verification_passed = verify_result.success
                if not verify_result.success:
                    result.success = True  # Install succeeded, verify failed
                    result.error_message = "Installation completed but verification failed"
                    self._log(f"Verification failed: {verify_cmd}")
                    return result
            else:
                result.verification_passed = True  # No verify command = assume ok

            result.success = True
            self._log("Installation completed successfully")
            return result

        except KeyboardInterrupt:
            result.error_message = "Installation interrupted by user (Ctrl+C)"
            self._log("INTERRUPTED by user")
            return result
        except PrivilegeError as e:
            result.error_message = str(e)
            self._log(f"PRIVILEGE ERROR: {e}")
            return result
        except InstallerError as e:
            result.error_message = str(e)
            self._log(f"INSTALLER ERROR: {e}")
            return result
        finally:
            if self.log_file_handle:
                self._log("--- end of log ---")
                self.log_file_handle.close()
                self.log_file_handle = None

    def _init_log(self):
        """Initialize the install log file."""
        log_dir = Path.home() / ".env-doctor"
        log_dir.mkdir(exist_ok=True)
        self.log_path = log_dir / "install.log"
        self.log_file_handle = open(self.log_path, "w")
        self._log(f"env-doctor cuda-install {self.cuda_version}")
        self._log(f"Platform: {self.platform_info.get('platform_key', 'unknown')}")
        self._log(f"CI: {self.ci_env.name}")
        self._log(f"Dry run: {self.dry_run}")

    def _log(self, line: str):
        """Write a timestamped line to the log file."""
        if self.log_file_handle:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_file_handle.write(f"[{timestamp}] {line}\n")
            self.log_file_handle.flush()

    def _check_privileges(self, steps):
        """
        Check if we have sufficient privileges to run the install steps.

        Raises PrivilegeError if insufficient.
        """
        needs_sudo = any(s.strip().startswith("sudo ") for s in steps)
        if not needs_sudo:
            return

        import platform as plat
        if plat.system() == "Windows":
            return  # Windows steps don't use sudo

        # On Linux/Mac, check if we're root
        if hasattr(os, 'geteuid') and os.geteuid() == 0:
            return

        # If geteuid not available (Windows), skip check
        if not hasattr(os, 'geteuid'):
            return

        # Not root — check if sudo is available non-interactively
        if self.yes or self.ci_env.is_ci:
            # In headless mode, try passwordless sudo
            try:
                result = subprocess.run(
                    ["sudo", "-n", "true"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    raise PrivilegeError(
                        "Installation requires root privileges. "
                        "Run with sudo or configure passwordless sudo for CI."
                    )
            except FileNotFoundError:
                raise PrivilegeError("sudo not found. Run as root.")
            except subprocess.TimeoutExpired:
                raise PrivilegeError("sudo timed out. Configure passwordless sudo for CI.")
        # Interactive mode — sudo will prompt for password, that's fine

    def _prompt_confirmation(self, steps):
        """
        Show installation steps and ask for confirmation.

        Returns True if user confirms, False otherwise.
        """
        print("\nThe following commands will be executed:")
        print("-" * 50)
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

        post_install = self.install_info.get("post_install", [])
        if post_install:
            print("\nPost-install environment setup:")
            for step in post_install:
                print(f"  - {step}")

        verify = self.install_info.get("verify")
        if verify:
            print(f"\nVerification: {verify}")

        print()
        try:
            response = input("Proceed with installation? [y/N] ").strip().lower()
            return response in ("y", "yes")
        except EOFError:
            return False

    def _execute_step(self, command, phase, step_num, total_steps):
        """
        Execute a single command step.

        Args:
            command: Shell command string to run
            phase: "install", "post_install", or "verify"
            step_num: Current step number (for display)
            total_steps: Total steps (for display)

        Returns:
            StepResult
        """
        prefix = "[DRY RUN] " if self.dry_run else ""
        print(f"\n{prefix}[{step_num}/{total_steps}] {command}")
        self._log(f"[{phase}] [{step_num}/{total_steps}] {command}")

        if self.dry_run:
            return StepResult(
                command=command,
                phase=phase,
                success=True,
                return_code=0,
                stdout="",
                stderr="",
                duration_seconds=0.0,
            )

        start = time.monotonic()
        stdout_lines = []
        stderr_lines = []

        try:
            # Use shell=True since commands may have pipes, redirects, etc.
            import platform as plat
            shell_exec = True
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Stream output in real-time
            # Read stdout and stderr
            for line in process.stdout:
                line_stripped = line.rstrip("\n")
                print(f"  {line_stripped}")
                self._log(f"  [stdout] {line_stripped}")
                stdout_lines.append(line_stripped)

            # Read any remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                for line in stderr_output.splitlines():
                    print(f"  [stderr] {line}", file=sys.stderr)
                    self._log(f"  [stderr] {line}")
                    stderr_lines.append(line)

            process.wait()
            duration = time.monotonic() - start

            success = process.returncode == 0
            if success:
                self._log(f"  -> OK ({duration:.1f}s)")
            else:
                self._log(f"  -> FAILED exit={process.returncode} ({duration:.1f}s)")

            return StepResult(
                command=command,
                phase=phase,
                success=success,
                return_code=process.returncode,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                duration_seconds=round(duration, 2),
            )

        except Exception as e:
            duration = time.monotonic() - start
            self._log(f"  -> EXCEPTION: {e}")
            return StepResult(
                command=command,
                phase=phase,
                success=False,
                return_code=-1,
                stdout="\n".join(stdout_lines),
                stderr=str(e),
                duration_seconds=round(duration, 2),
            )

    def _persist_env_vars(self):
        """
        Parse post_install export commands and persist them using the
        CI-appropriate method.

        Returns:
            Dict of env var name -> value that were persisted
        """
        post_install = self.install_info.get("post_install", [])
        env_vars = {}

        # Parse export commands from post_install
        for line in post_install:
            match = re.match(r'^export\s+(\w+)=(.+)$', line.strip())
            if match:
                var_name = match.group(1)
                var_value = match.group(2)
                # Expand ${VAR:+:$VAR} patterns for current env
                var_value = self._expand_env_value(var_name, var_value)
                env_vars[var_name] = var_value

        if not env_vars:
            return {}

        if self.dry_run:
            print("\n[DRY RUN] Would persist environment variables:")
            for name, value in env_vars.items():
                print(f"  {name}={value}")
            self._log(f"[DRY RUN] Would persist env vars: {env_vars}")
            return env_vars

        method = self.ci_env.env_persist_method
        self._log(f"Persisting env vars via {method}: {list(env_vars.keys())}")

        if method == "github_env":
            self._persist_github_env(env_vars)
        elif method == "circleci_bash_env":
            self._persist_circleci_env(env_vars)
        elif method == "azure_vso":
            self._persist_azure_env(env_vars)
        elif method == "shell_rc":
            self._persist_shell_rc(env_vars)
        elif method == "windows_setx":
            self._persist_windows_setx(env_vars)
        else:
            # gitlab_dotenv, export_echo, jenkins — just echo
            self._persist_echo(env_vars)

        # Also set in current process
        for name, value in env_vars.items():
            os.environ[name] = value

        return env_vars

    def _expand_env_value(self, var_name, raw_value):
        """Expand shell-style ${VAR:+:$VAR} in env values."""
        # Handle pattern like /usr/local/cuda-12.8/bin${PATH:+:${PATH}}
        # Replace ${VAR:+:${VAR}} with :current_value if VAR is set
        def replace_conditional(m):
            env_var = m.group(1)
            current = os.environ.get(env_var, "")
            if current:
                return f":{current}"
            return ""

        result = re.sub(r'\$\{\w+:\+:\$\{(\w+)\}\}', replace_conditional, raw_value)
        return result

    def _persist_github_env(self, env_vars):
        """Write env vars to $GITHUB_ENV and PATH additions to $GITHUB_PATH."""
        env_file = self.ci_env.github_env_file
        path_file = self.ci_env.github_path_file

        for name, value in env_vars.items():
            if name == "PATH" and path_file:
                # Extract the new path component (before the existing PATH)
                new_paths = value.split(os.pathsep)
                try:
                    with open(path_file, "a") as f:
                        for p in new_paths:
                            if p and p != os.environ.get("PATH", ""):
                                f.write(f"{p}\n")
                    print(f"  Added to $GITHUB_PATH")
                    self._log(f"  Wrote PATH entries to {path_file}")
                except IOError as e:
                    self._log(f"  Failed to write GITHUB_PATH: {e}")
            elif env_file:
                try:
                    with open(env_file, "a") as f:
                        f.write(f"{name}={value}\n")
                    print(f"  Set {name} in $GITHUB_ENV")
                    self._log(f"  Wrote {name} to {env_file}")
                except IOError as e:
                    self._log(f"  Failed to write GITHUB_ENV: {e}")

    def _persist_circleci_env(self, env_vars):
        """Append export commands to $BASH_ENV for CircleCI."""
        bash_env = os.environ.get("BASH_ENV", "")
        if not bash_env:
            self._persist_echo(env_vars)
            return

        try:
            with open(bash_env, "a") as f:
                for name, value in env_vars.items():
                    f.write(f"export {name}={value}\n")
            print(f"  Wrote exports to $BASH_ENV ({bash_env})")
            self._log(f"  Wrote to BASH_ENV: {bash_env}")
        except IOError as e:
            self._log(f"  Failed to write BASH_ENV: {e}")
            self._persist_echo(env_vars)

    def _persist_azure_env(self, env_vars):
        """Use Azure Pipelines ##vso syntax."""
        for name, value in env_vars.items():
            print(f"##vso[task.setvariable variable={name}]{value}")
            self._log(f"  Azure vso: {name}")

    def _persist_shell_rc(self, env_vars):
        """Append export commands to ~/.bashrc or ~/.zshrc."""
        shell = os.environ.get("SHELL", "/bin/bash")
        if "zsh" in shell:
            rc_file = Path.home() / ".zshrc"
        else:
            rc_file = Path.home() / ".bashrc"

        marker = f"# Added by env-doctor cuda-install ({self.cuda_version})"
        lines = [f"\n{marker}\n"]
        for name, value in env_vars.items():
            lines.append(f"export {name}={value}\n")

        if self.dry_run:
            print(f"\n[DRY RUN] Would append to {rc_file}:")
            for line in lines:
                print(f"  {line.rstrip()}")
            return

        try:
            with open(rc_file, "a") as f:
                f.writelines(lines)
            print(f"  Appended exports to {rc_file}")
            self._log(f"  Wrote to {rc_file}")
        except IOError as e:
            self._log(f"  Failed to write {rc_file}: {e}")
            self._persist_echo(env_vars)

    def _persist_windows_setx(self, env_vars):
        """Use setx for Windows env var persistence (except PATH, handled by winget)."""
        for name, value in env_vars.items():
            if name == "PATH":
                # winget handles PATH for Windows CUDA installs
                print(f"  PATH is managed by the installer")
                continue
            try:
                result = subprocess.run(
                    ["setx", name, value],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    print(f"  Set {name} via setx")
                    self._log(f"  setx {name} OK")
                else:
                    self._log(f"  setx {name} failed: {result.stderr}")
            except Exception as e:
                self._log(f"  setx {name} exception: {e}")

    def _persist_echo(self, env_vars):
        """Fallback: print export commands for user to copy."""
        print("\n  Environment variables to set:")
        for name, value in env_vars.items():
            print(f"  export {name}={value}")
        self._log("  Printed export commands (echo fallback)")
