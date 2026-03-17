# cuda-install

Get step-by-step CUDA Toolkit installation instructions tailored to your platform.

## Overview

The `cuda-install` command automatically:

1. **Detects your platform** (OS, distribution, version, architecture, WSL2)
2. **Analyzes your GPU driver** to determine the best CUDA version
3. **Provides copy-paste installation commands** specific to your system
4. **Includes post-installation setup** (environment variables, verification)
5. **Shows official download links** for manual installation if needed

## Usage

### Auto-detect from GPU driver (recommended)

```bash
env-doctor cuda-install
```

This will:
- Check your NVIDIA driver version
- Recommend the best CUDA Toolkit version
- Show platform-specific installation steps

### Install specific CUDA version

```bash
env-doctor cuda-install 12.8
env-doctor cuda-install 12.4
env-doctor cuda-install 12.1
env-doctor cuda-install 11.8
```

### Execute the installation

```bash
# Preview what would run (no changes made)
env-doctor cuda-install --dry-run

# Interactive install (asks [y/N] before running)
env-doctor cuda-install --run

# Headless install — skip confirmation (for CI/scripts)
env-doctor cuda-install --run --yes

# Specific version, headless
env-doctor cuda-install 12.6 --run --yes
```

### Get JSON output (for automation/CI)

```bash
env-doctor cuda-install --json
env-doctor cuda-install 12.6 --json

# JSON output from a real install run
env-doctor cuda-install --run --yes --json
```

Returns structured JSON with platform info, recommended version, and install steps for machine processing.

## Example Output

### Linux (Ubuntu)

```bash
$ env-doctor cuda-install
```

```
============================================================
CUDA TOOLKIT INSTALLATION GUIDE
============================================================

Detected Platform:
    Linux (ubuntu 22.04, x86_64)

Driver: 535.146.02 (supports up to CUDA 12.2)
Recommended CUDA Toolkit: 12.1

============================================================
Ubuntu 22.04 (x86_64) - Network Install
============================================================

Installation Steps:
------------------------------------------------------------
    1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    2. sudo dpkg -i cuda-keyring_1.1-1_all.deb
    3. sudo apt-get update
    4. sudo apt-get -y install cuda-toolkit-12-1

Post-Installation Setup:
------------------------------------------------------------
    export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    TIP: Add the above exports to ~/.bashrc or ~/.zshrc

Verify Installation:
------------------------------------------------------------
    nvcc --version

Official Download Page:
    https://developer.nvidia.com/cuda-12-1-0-download-archive

============================================================
After installation, run 'env-doctor check' to verify.
============================================================
```

### WSL2 (Special Instructions)

```bash
$ env-doctor cuda-install
```

```
============================================================
CUDA TOOLKIT INSTALLATION GUIDE
============================================================

Detected Platform:
    WSL2 (ubuntu 22.04)

Driver: 560.35.03 (supports up to CUDA 12.6)
Recommended CUDA Toolkit: 12.6

============================================================
WSL2 (Ubuntu) - DO NOT install driver inside WSL
============================================================

Prerequisites:
    - Ensure Windows NVIDIA driver >= 560.xx is installed on the HOST
    - DO NOT install nvidia-driver packages inside WSL2

Installation Steps:
------------------------------------------------------------
    1. wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    2. sudo dpkg -i cuda-keyring_1.1-1_all.deb
    3. sudo apt-get update
    4. sudo apt-get -y install cuda-toolkit-12-6

Post-Installation Setup:
------------------------------------------------------------
    export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    TIP: Add the above exports to ~/.bashrc or ~/.zshrc

Verify Installation:
------------------------------------------------------------
    nvcc --version

Notes:
    GPU driver is forwarded from Windows host. Only install the toolkit inside WSL2.

============================================================
```

### Windows

```bash
$ env-doctor cuda-install
```

```
============================================================
CUDA TOOLKIT INSTALLATION GUIDE
============================================================

Detected Platform:
    Windows (x86_64)

Driver: 560.35.03 (supports up to CUDA 12.6)
Recommended CUDA Toolkit: 12.6

============================================================
Windows 10/11 (x86_64) - winget
============================================================

Installation Steps:
------------------------------------------------------------
    1. winget install Nvidia.CUDA --version 12.6

Post-Installation Setup:
------------------------------------------------------------
    The installer automatically adds CUDA to PATH
    Restart your terminal/IDE after installation

Verify Installation:
------------------------------------------------------------
    nvcc --version

Notes:
    If winget is not available, download manually from the official page.

Official Download Page:
    https://developer.nvidia.com/cuda-12-6-0-download-archive

============================================================
```

**New in 0.2.6**: Windows now uses `winget` (Windows Package Manager) for automated installation instead of manual GUI steps.

## Supported Platforms

### Linux Distributions

| Distribution | Versions | Package Manager |
|--------------|----------|-----------------|
| Ubuntu | 18.04, 20.04, 22.04, 24.04 | apt (deb) |
| Debian | 11, 12 | apt (deb) |
| RHEL / Rocky / AlmaLinux | 7, 8, 9 | dnf/yum (rpm) |
| Fedora | 39+ | dnf (rpm) |
| WSL2 Ubuntu | All versions | apt (deb) |

### Other Platforms

| Platform | Installation Method |
|----------|---------------------|
| Windows 10/11 | winget (Windows Package Manager) |
| macOS | Not supported (CUDA deprecated) |
| Conda (any platform) | `conda install cuda-toolkit` |

**New in 0.2.6**: Windows now uses `winget` for automated installation instead of manual GUI steps.

## CUDA Versions Supported

The tool provides installation instructions for 9 CUDA versions:

**CUDA 12.x (Latest)**
- **CUDA 12.8** (Latest, requires driver >= 570.xx)
- **CUDA 12.6** (Latest stable, requires driver >= 560.xx)
- **CUDA 12.5** (requires driver >= 555.xx)
- **CUDA 12.4** (TensorFlow 2.16+, requires driver >= 550.xx)
- **CUDA 12.2** (requires driver >= 535.xx)
- **CUDA 12.1** (PyTorch 2.x sweet spot, requires driver >= 530.xx)
- **CUDA 12.0** (First CUDA 12 release, requires driver >= 525.xx)

**CUDA 11.x (Legacy)**
- **CUDA 11.8** (Latest CUDA 11, requires driver >= 520.xx)
- **CUDA 11.7** (requires driver >= 515.xx)

## Version Recommendation Logic

The tool automatically recommends the best CUDA version based on:

1. **Your GPU driver** - Maps driver version to max supported CUDA
2. **Forward compatibility** - Recommends latest stable CUDA your driver supports
3. **Library compatibility** - Considers PyTorch/TensorFlow requirements

**Example Mappings (Updated in 0.2.6):**

| Driver Version | Max CUDA | Recommended Toolkit |
|----------------|----------|---------------------|
| 570.xx+ | 12.8 | CUDA 12.8 |
| 560.xx | 12.6 | CUDA 12.6 |
| 555.xx | 12.5 | CUDA 12.5 |
| 550.xx | 12.4 | CUDA 12.4 |
| 535.xx | 12.2 | CUDA 12.2 |
| 530.xx | 12.1 | CUDA 12.1 |
| 525.xx | 12.0 | CUDA 12.0 |
| 520.xx | 11.8 | CUDA 11.8 |
| 515.xx | 11.7 | CUDA 11.7 |

## Executing the Installation

### Interactive Mode

```bash
env-doctor cuda-install --run
```

Shows all commands, asks `[y/N]`, then executes sequentially with real-time output. Stops on first failure.

### Headless / CI Mode

```bash
env-doctor cuda-install --run --yes
```

Skips confirmation. In `--yes` mode without root on Linux, requires passwordless sudo (`sudo -n true`); if that fails, aborts with a clear message rather than hanging.

### Dry Run

```bash
env-doctor cuda-install --dry-run
```

Prints every command prefixed with `[DRY RUN]`, goes through the full flow (including verification), but makes no changes to the system. Always writes a log.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Installation succeeded and `nvcc --version` passed |
| `1` | An installation step failed |
| `2` | Steps ran but verification (`nvcc --version`) failed |

### Install Log

Every `--run` or `--dry-run` writes a full timestamped log to:

- **Linux/macOS:** `~/.env-doctor/install.log`
- **Windows:** `%USERPROFILE%\.env-doctor\install.log`

The log is **overwritten each run** (not appended), so one log = one install attempt. The path is printed at the end of the run.

### JSON Output from `--run`

```bash
env-doctor cuda-install --run --yes --json
```

Returns an `InstallResult` JSON object:

```json
{
  "success": true,
  "cuda_version": "12.6",
  "platform_key": "linux_ubuntu_22.04_x86_64",
  "steps_completed": [
    {"command": "sudo apt-get update", "phase": "install", "success": true, "return_code": 0, "duration_seconds": 8.2}
  ],
  "steps_remaining": [],
  "env_vars_set": {"PATH": "/usr/local/cuda-12.6/bin:...", "LD_LIBRARY_PATH": "..."},
  "verification_passed": true,
  "error_message": null,
  "log_file": "/home/user/.env-doctor/install.log"
}
```

---

## Common Use Cases

### Case 1: New Machine Setup

```bash
# 1. Check current state
env-doctor check

# 2. Preview what will be installed
env-doctor cuda-install --dry-run

# 3. Execute the installation
env-doctor cuda-install --run

# 4. Verify
env-doctor check
```

### Case 2: Upgrade CUDA for PyTorch Compatibility

```bash
# 1. Check current versions
env-doctor check

# 2. Install specific version directly
env-doctor cuda-install 12.4 --run --yes

# 3. Verify
env-doctor check
```

### Case 3: WSL2 GPU Setup

```bash
# Inside WSL2:
# 1. Check if driver is forwarded
env-doctor check

# 2. Preview WSL2-specific instructions
env-doctor cuda-install --dry-run

# 3. Follow WSL2 prerequisites carefully (don't install driver in WSL!)
env-doctor cuda-install --run
```

### Case 4: CI/CD Pipeline

```yaml
# GitHub Actions
- name: Install CUDA
  run: env-doctor cuda-install --run --yes

# Capture JSON result for downstream steps
- name: Install CUDA (JSON)
  run: env-doctor cuda-install --run --yes --json > cuda_result.json
```

## JSON Output for Automation

Get structured JSON output for integration with CI/CD pipelines:

```bash
env-doctor cuda-install --json
```

Response format:
```json
{
  "platform": {
    "os": "windows",
    "distro": "windows",
    "arch": "x86_64",
    "is_wsl2": false
  },
  "recommended_version": "12.6",
  "driver_version": "560.35.03",
  "max_cuda": "12.6",
  "install_info": {
    "method": "winget",
    "label": "Windows 10/11 (x86_64) - winget",
    "steps": ["winget install Nvidia.CUDA --version 12.6"],
    "post_install": ["The installer automatically adds CUDA to PATH"],
    "verify": "nvcc --version",
    "download_page": "https://developer.nvidia.com/cuda-12-6-0-download-archive"
  }
}
```

**New in 0.2.6**: `--json` flag allows machine-readable output for automation, testing, and CI/CD integration.

## Post-Installation

After installing CUDA Toolkit:

1. **Verify installation:**
   ```bash
   nvcc --version
   ```

2. **Run full diagnostic:**
   ```bash
   env-doctor check
   ```

3. **Install Python libraries:**
   ```bash
   env-doctor install torch
   ```

## Troubleshooting

### "No NVIDIA driver detected"

Install the NVIDIA driver first:
- Linux: Use your distribution's package manager or NVIDIA's .run installer
- Windows: Download from [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
- WSL2: Install driver on Windows host, not in WSL2

### "No specific instructions for your platform"

The tool will show:
- List of available platforms for your CUDA version
- Generic download link to NVIDIA's website
- Conda installation as universal fallback

You can also specify a different CUDA version that might have your platform:
```bash
env-doctor cuda-install 12.1
```

### Multiple CUDA Installations

If you already have CUDA installed:
- The new installation will be in a versioned directory (e.g., `/usr/local/cuda-12.4`)
- Update `CUDA_HOME` and `PATH` to point to the version you want
- Use `env-doctor cuda-info` to see all installations

## Integration with Other Commands

The `cuda-install` command works seamlessly with:

- `env-doctor check` - See what's currently installed
- `env-doctor cuda-info` - Detailed analysis of CUDA installations
- `env-doctor install <lib>` - Get library install commands after CUDA is set up

## See Also

- [`check`](check.md) - Diagnose your environment
- [`cuda-info`](cuda-info.md) - Detailed CUDA toolkit analysis
- [`install`](install.md) - Get safe library install commands
- [WSL2 GPU Guide](../guides/wsl2.md) - Complete WSL2 setup guide
