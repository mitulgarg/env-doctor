# Changelog

All notable changes to env-doctor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`--recommend` flag for `env-doctor model`**: Suggests cloud GPU instances (AWS, GCP, Azure) sorted by cost when a model doesn't fit locally
- **`--vram` flag**: Direct VRAM-based cloud instance lookup without specifying a model name (`env-doctor model --vram 80000 --recommend`)
- **`cloud_instances.json`**: Static data file covering 17 GPU instances across AWS, GCP, and Azure
- **Cloud recommendations via MCP**: `model_check` tool accepts `recommend` boolean to include cloud instance suggestions in results

## [0.2.8] - 2026-03-18

### Added
- **uv Support**: `uv tool install env-doctor` and `uvx env-doctor` now work as first-class install paths alongside `pip install env-doctor`
  - Added `[dependency-groups]` with `dev = ["pytest"]` so contributors can run `uv sync --group dev` to get a full dev environment
  - `uv.lock` committed for reproducible contributor environments
  - Documented `uv tool install` and `uvx` in README and `docs/getting-started.md`

### Fixed
- **Wheel missing data files**: `data/*.json` (compatibility, model requirements, compute capability, etc.) were included in sdist via `MANIFEST.in` but not in wheels. Added `[tool.setuptools.package-data]` to fix this for all install methods.

### Changed
- Bumped version to 0.2.8

## [0.2.7] - 2026-03-06

### Added
- **CUDA Auto-Installer `--run` flag**: Execute CUDA Toolkit installation directly instead of just printing instructions
  - `--run` — runs steps interactively, prompting `[y/N]` before execution
  - `--yes` — headless/non-interactive mode, auto-confirms all steps (CI-friendly)
  - `--dry-run` — preview all commands without executing anything
  - Timestamped log written to `~/.env-doctor/install.log` on every run
  - Exit codes: `0` success, `1` step failed, `2` installed but verification failed

### Fixed
- CI env var isolation: `test_ci` no longer reads real GitHub Actions env vars, making it safe to run in any CI environment

### Changed
- Bumped version to 0.2.7

## [0.2.6] - 2026-03-06

### Added
- **Expanded CUDA Toolkit Version Coverage**: Added 5 new CUDA versions (12.8, 12.5, 12.2, 12.0, 11.7) with full platform-specific installation instructions
  - Total of 9 supported CUDA versions (up from 4)
  - Platform-appropriate keyring versions and distro support
  - Covers Ubuntu 18/20/22/24, Debian 11/12, RHEL 7/8/9, Fedora 39+, WSL2, Windows, and Conda

- **Windows Installation: GUI -> winget Commands**
  - Replaced multi-step GUI installer instructions with single `winget install Nvidia.CUDA --version X.Y` command
  - Added fallback `download_page` field for winget unavailability
  - Cleaner, more automated installation experience

- **JSON Output for cuda-install Command**: New `--json` flag for the `cuda-install` command
  - Structured JSON output matching MCP tool format
  - Includes platform detection, recommended version, driver version, and install instructions
  - Error paths also return JSON for consistent machine-readable output
  - Useful for automation and CI/CD pipelines

- **Version Recommendation Map Improvements**
  - Each version now maps to itself for better accuracy
  - In-between versions map to the nearest available version below driver max CUDA
  - 11.0-11.6 now correctly map to 11.7 (previously 11.8)

### Changed
- Bumped version to 0.2.6

### Fixed
- **Cross-Platform Test Fixes**
  - `test_scan_json_output`: Changed hardcoded `/tmp` to `tempfile.gettempdir()` for Windows compatibility
  - `test_default_output_not_json`: Added UTF-8 encoding and `PYTHONUTF8=1` to handle emoji output on Windows

### Tests
- Added `test_cuda_install_json_output` - subprocess test for `cuda-install --json`
- Added `test_json_output_windows` - verifies JSON with winget method
- Added `test_json_output_no_driver` - verifies JSON error handling
- Updated `test_windows_installation` - asserts winget command
- Updated `test_auto_detect_ubuntu_with_driver` - 12.2 now recommends 12.2
- Fixed 2 pre-existing platform-specific test failures

## [0.2.4] - 2026-02-20

### Added
- **Hard vs Soft GPU Architecture Mismatch Detection**: The `check` command now distinguishes between two types of architecture mismatch:
  - Hard failure (`cuda_available=False` or `None`): shown as `❌ ARCHITECTURE MISMATCH` — `torch.cuda.is_available()` likely returns `False`
  - Soft failure (`cuda_available=True`): shown as `⚠️ ARCHITECTURE MISMATCH (Soft)` — PTX JIT allowed CUDA to initialise but complex ops may fail
- `cuda_available` field included in `--json` / `--ci` output (status remains `"mismatch"` in both cases for backward compatibility)
- 7 new unit tests covering hard, soft, `None→hard`, compatible, and JSON field assertions

### Changed
- `check_compute_capability_compatibility()` now accepts a `cuda_available` parameter to determine failure mode
- Internal helper `_get_torch_cuda_available()` safely probes `torch` at runtime

### Documentation
- Updated `docs/commands/check.md` with hard/soft mismatch examples and CI/CD guidance

## [0.2.3] - 2026-02-20

### Added
- **GPU Compute Capability vs PyTorch Architecture Mismatch Detection**: New check in the `check` command that detects when a GPU's SM architecture is not covered by the installed PyTorch wheel
  - Catches the silent failure where `torch.cuda.is_available()` returns `False` even when the driver and CUDA toolkit are healthy (e.g. RTX 5070 / Blackwell with stable PyTorch)
  - PTX forward-compatibility awareness
- **New Data File**: `data/compute_capability.json` — CC-to-SM mapping from Kepler through Blackwell
- **New Detector**: `detectors/compute_capability.py` — helpers for SM lookup, architecture name resolution, and `is_sm_in_arch_list()`
- Extended `NvidiaDriverDetector` to capture `compute_capability` per GPU via `nvmlDeviceGetCudaComputeCapability()` and `nvidia-smi`
- Extended `PythonLibraryDetector` to capture `torch.cuda.get_arch_list()` as `arch_list` in metadata
- `compute_compatibility` included in `--json` / `--ci` output
- 21 unit tests for compute capability utilities; 2 unit tests for `arch_list` detection

### Documentation
- Updated `README.md` features table and diagnosis example
- Added new section, example outputs, and JSON schema to `docs/commands/check.md`

## [0.2.2] - 2026-02-18

### Added
- **New Command**: `python-compat` — checks Python version compatibility with ML libraries (PyTorch, TensorFlow, JAX, etc.)
  - Compares installed Python version against each library's supported range
  - Flags incompatible or end-of-life Python versions with actionable recommendations
- **New Data File**: `data/python_compatibility.json` — compatibility matrix for major ML libraries
- **New Detector**: `detectors/python_compat.py`
- **New MCP Tool**: `python_compat` — exposes Python compatibility check via MCP server
- New `python_compat` query added to `db.py`
- Integration and unit test suites for the new detector and CLI path (12 new files, 1 035+ lines)

### Documentation
- Added `docs/commands/python-compat.md` — full command reference with examples
- Updated `docs/index.md` to list the new command
- Updated `README.md` with `python-compat` feature and usage examples
- Updated MCP documentation to reflect all 10 tools

## [0.2.1] - 2026-02-12

### Documentation
- Enhanced `getting-started.md` with CUDA Toolkit installation guide section
- Added `cuda-install` command to Quick Command Reference table
- Updated version to 0.2.1 for feature release

### Changed
- Minor improvements and documentation refinements for `cuda-install` feature

## [0.2.0] - 2026-02-09

### Added
- **New Command**: `cuda-install` - Step-by-step CUDA Toolkit installation guide
  - Auto-detects platform (Ubuntu, Debian, RHEL, Fedora, WSL2, Windows, macOS)
  - Recommends CUDA version based on GPU driver
  - Provides copy-paste installation commands for each platform
  - Includes post-installation setup and verification steps
  - Supports CUDA 12.6, 12.4, 12.1, and 11.8

- **New Data File**: `cuda_toolkit_install.json`
  - Platform-specific installation instructions for 7+ platforms
  - Version recommendation mapping based on driver capabilities
  - Over 500 lines of curated installation data

- **New Utility**: `platform_detect.py`
  - Detects OS, distribution, version, and architecture
  - WSL2 detection support
  - Fallback detection mechanisms

- **5 New MCP Tools**:
  - `cuda_install` - Get CUDA installation instructions via MCP
  - `install_command` - Get safe pip install commands for libraries
  - `cuda_info` - Detailed CUDA toolkit analysis
  - `cudnn_info` - Detailed cuDNN library analysis
  - `docker_compose_validate` - Validate docker-compose.yml for GPU config

- **MCP Testing Suite**:
  - `test_mcp_tools.py` - Automated test for all 10 MCP tools
  - `test_mcp_interactive.py` - Interactive MCP tool tester
  - `test_mcp_manual.sh` - Manual JSON-RPC testing
  - Comprehensive documentation in `docs/MCP_TESTING.md`

### Changed
- **Enhanced CUDA Toolkit Detector**: Now provides targeted recommendations when CUDA not found
  - Detects GPU driver and recommends specific CUDA version
  - Suggests running `cuda-install` command for step-by-step instructions
  - Falls back to generic recommendations if driver not detected

- **Improved Recommendations**: All NOT_FOUND messages now include actionable next steps

### Documentation
- Added `docs/QUICK_START_MCP.md` - Quick reference for MCP testing
- Added `docs/MCP_TESTING.md` - Complete MCP integration guide
- Added `tests/README_MCP_TESTS.md` - Test suite documentation
- Updated CLI help examples to include `cuda-install` command

## [0.1.4] - 2024-XX-XX

### Added
- Model compatibility checking with VRAM analysis
- Dockerfile and docker-compose validation
- cuDNN detection and analysis
- WSL2 environment detection
- MCP server integration (5 initial tools)

### Changed
- Refactored to detector-based architecture
- Improved error messages and recommendations

## [0.1.3] - 2024-XX-XX

### Added
- Python library detection (PyTorch, TensorFlow, JAX)
- CUDA toolkit comprehensive detection
- Driver compatibility validation

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- NVIDIA driver detection
- Basic CUDA toolkit detection
- Library installation commands

[0.2.8]: https://github.com/mitulgarg/env-doctor/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/mitulgarg/env-doctor/compare/v0.2.6...v0.2.7
[0.2.4]: https://github.com/mitulgarg/env-doctor/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/mitulgarg/env-doctor/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mitulgarg/env-doctor/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mitulgarg/env-doctor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mitulgarg/env-doctor/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/mitulgarg/env-doctor/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/mitulgarg/env-doctor/compare/v0.1.0...v0.1.3
[0.1.0]: https://github.com/mitulgarg/env-doctor/releases/tag/v0.1.0
