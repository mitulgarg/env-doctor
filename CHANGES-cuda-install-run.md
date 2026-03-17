# Changes: Automatic CUDA Installation Execution

**Branch:** `feat/cuda-install-run`
**Date:** 2026-03-18

---

## Summary

The `env-doctor cuda-install` command now goes beyond displaying steps — it can actually **execute the installation** with `--run`, skip prompts for CI with `--yes`, or preview execution without side effects using `--dry-run`.

---

## New Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--run` | | Execute the installation steps |
| `--yes` | `-y` | Skip confirmation prompts (headless/CI) |
| `--dry-run` | | Print commands with `[DRY RUN]` prefix, no execution |

### Usage examples

```bash
env-doctor cuda-install                    # Display only (unchanged default)
env-doctor cuda-install --dry-run          # Preview what would run
env-doctor cuda-install --run              # Interactive install (asks [y/N])
env-doctor cuda-install --run --yes        # Headless CI install
env-doctor cuda-install --run --json       # Machine-readable result
env-doctor cuda-install 12.6 --run --yes   # Specific version, headless
```

---

## New Files

### `src/env_doctor/ci.py`

CI environment detection. Exports `detect_ci_environment()` returning a `CIEnvironment` dataclass.

**Detected systems and their env-var persistence methods:**

| CI System | Env Var | Persistence |
|-----------|---------|-------------|
| GitHub Actions | `GITHUB_ACTIONS=true` | Write to `$GITHUB_ENV` / `$GITHUB_PATH` |
| GitLab CI | `GITLAB_CI=true` | Echo export commands (dotenv artifact) |
| CircleCI | `CIRCLECI=true` | Append to `$BASH_ENV` |
| Azure Pipelines | `TF_BUILD=True` | `##vso[task.setvariable]` syntax |
| Jenkins | `JENKINS_URL` set | Echo export commands |
| Generic CI | `CI=true` | Echo export commands |
| Local Linux | — | Append to `~/.bashrc` or `~/.zshrc` |
| Local Windows | — | `setx` (PATH handled by winget) |

### `src/env_doctor/installer.py`

Execution engine. Core class: `CudaInstaller`.

**What it does:**

1. `_check_privileges()` — verifies sudo/admin rights before running; on Linux checks `geteuid`, tries `sudo -n true` in headless mode; Windows checks `ctypes.windll.shell32.IsUserAnAdmin()`
2. `_prompt_confirmation()` — shows all commands and asks `[y/N]`; skipped with `--yes`
3. Sequential subprocess execution via `Popen` with real-time stdout/stderr streaming; stops on first failure and records `steps_remaining`
4. `_persist_env_vars()` — parses `export` lines from `post_install`, persists via CI-appropriate method
5. Verification — runs `install_info["verify"]` command
6. Returns `InstallResult` with `.to_dict()` for JSON output

**Dataclasses:**

```python
@dataclass
class StepResult:
    command: str
    phase: str          # "install" | "verify"
    success: bool
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float

@dataclass
class InstallResult:
    success: bool
    cuda_version: str
    platform_key: str
    steps_completed: list[StepResult]
    steps_remaining: list[str]
    env_vars_set: dict[str, str]
    verification_passed: bool
    error_message: Optional[str]
    log_file: Optional[str]
```

**Install log:** Every run (including dry-run) writes a full timestamped log to `~/.env-doctor/install.log` (overwritten each run). Path is reported at end of output.

---

## Modified Files

### `src/env_doctor/core/exceptions.py`

Added:

```python
class InstallerError(Exception): ...
class PrivilegeError(InstallerError): ...
class StepExecutionError(InstallerError):
    def __init__(self, step_index, command, return_code, stderr): ...
```

### `src/env_doctor/cli.py`

- Added `--run`, `--yes`/`-y`, `--dry-run` to `cuda-install` argparse subcommand
- Updated `cuda_install_command()` signature to accept `run`, `dry_run`, `yes`
- After `install_info` resolution: if `run` or `dry_run`, instantiates `CudaInstaller` and handles `InstallResult`; otherwise falls through to existing display-only behavior (unchanged)
- Exit codes: `0` = success, `1` = install failed, `2` = install ok but verification failed
- Added `description=` to `dockerfile` and `docker-compose` parsers so GPU/CUDA appears in `--help`

### `tests/integration/test_container_cli.py`

Two pre-existing bugs fixed:

1. **Windows encoding crash** — `subprocess.run(..., text=True)` used `cp1252` which can't decode emoji bytes. Fixed by adding `encoding="utf-8", errors="replace"` to all subprocess calls.
2. **`cwd="/tmp"` on Windows** — `/tmp` doesn't exist on Windows. Fixed by using `tempfile.gettempdir()`.

---

## New Tests

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_ci.py` | 11 | All 8 CI systems, priority order, local Linux/Windows |
| `tests/unit/test_installer.py` | 17 | Happy path, step failure, verification failure, privilege checks, confirmation prompt, `--yes` bypass, Ctrl+C, log file, `InstallResult.to_dict()`, env persistence |
| `tests/integration/test_cuda_install_run.py` | 8 | Default behavior unchanged, dry-run Ubuntu/Windows, `--run` success/failure exit codes, `--run --json`, specific version |

**Total: 36 new tests, all passing.**

---

## Exit Codes (`--run` mode)

| Code | Meaning |
|------|---------|
| `0` | Installation succeeded and verification passed |
| `1` | An installation step failed |
| `2` | Steps completed but `nvcc --version` (verify) failed |
| `130` | Interrupted with Ctrl+C |

---

## Design Decisions

- **`--run` is opt-in.** Default is display-only — no accidental installs.
- **Stop on first failure, no rollback.** Package manager steps are idempotent; re-running is safe.
- **Always log.** `~/.env-doctor/install.log` is always written, regardless of `--json` or verbosity, for post-mortem debugging.
- **CI-aware env persistence.** Each CI system has the right method — GitHub writes to `$GITHUB_ENV`, CircleCI appends to `$BASH_ENV`, Azure emits `##vso` syntax, local Linux appends to shell rc, etc.
