# CI/CD Integration

Env-Doctor supports JSON output and proper exit codes for seamless CI/CD integration.

## Quick Start

```yaml
# GitHub Actions
- run: pip install env-doctor
- run: env-doctor check --ci
```

## Exit Codes

When using `--ci` flag:

| Code | Meaning | Action |
|------|---------|--------|
| `0` | All checks passed | Continue pipeline |
| `1` | Warnings or non-critical issues | Review, may continue |
| `2` | Critical errors detected | Fail pipeline |

## GitHub Actions

### Basic Validation

```yaml
name: Validate Environment
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install env-doctor
      - run: env-doctor check --ci
```

### GPU Runner Validation

For self-hosted GPU runners:

```yaml
name: GPU Environment Check
on: [push]

jobs:
  gpu-check:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: pip install env-doctor
      - name: Validate GPU Environment
        run: env-doctor check --ci
      - name: Check Model Compatibility
        run: env-doctor model llama-3-8b --json
```

### Dockerfile Validation

```yaml
name: Validate Docker
on:
  push:
    paths:
      - 'Dockerfile'
      - 'docker-compose.yml'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install env-doctor
      - name: Validate Dockerfile
        run: env-doctor dockerfile --ci
      - name: Validate Docker Compose
        run: env-doctor docker-compose --ci
```

## GitLab CI

```yaml
stages:
  - validate

validate-environment:
  stage: validate
  image: python:3.10
  script:
    - pip install env-doctor
    - env-doctor check --ci
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

## JSON Output

For custom parsing and reporting:

```bash
env-doctor check --json
```

### Parsing in Bash

```bash
#!/bin/bash

# Run check and capture output
RESULT=$(env-doctor check --json)

# Extract specific fields with jq
STATUS=$(echo "$RESULT" | jq -r '.status')
DRIVER_VERSION=$(echo "$RESULT" | jq -r '.checks.driver.version')
CUDA_VERSION=$(echo "$RESULT" | jq -r '.checks.cuda.version')

echo "Status: $STATUS"
echo "Driver: $DRIVER_VERSION"
echo "CUDA: $CUDA_VERSION"

# Conditional logic
if [ "$STATUS" = "error" ]; then
  echo "Environment check failed!"
  exit 1
fi

# Detect hard vs soft GPU architecture mismatch
ARCH_STATUS=$(echo "$RESULT" | jq -r '.checks.compute_compatibility.status // empty')
CUDA_AVAILABLE=$(echo "$RESULT" | jq -r '.checks.compute_compatibility.cuda_available // empty')

if [ "$ARCH_STATUS" = "mismatch" ]; then
  if [ "$CUDA_AVAILABLE" = "false" ]; then
    echo "Hard mismatch: torch.cuda.is_available() is False — install PyTorch nightly"
    exit 1
  elif [ "$CUDA_AVAILABLE" = "true" ]; then
    echo "Soft mismatch: CUDA works via PTX JIT but performance may degrade"
  fi
fi
```

### Parsing in Python

```python
import json
import subprocess

result = subprocess.run(
    ["env-doctor", "check", "--json"],
    capture_output=True,
    text=True
)
data = json.loads(result.stdout)

# Access check results
if data["status"] == "success":
    print(f"Driver: {data['checks']['driver']['version']}")
    print(f"CUDA: {data['checks']['cuda']['version']}")
else:
    print(f"Issues found: {data['summary']['issues_count']}")
```

## Automated CUDA Installation

The `--run --yes` flags let you execute CUDA Toolkit installation directly from CI pipelines:

```yaml
# GitHub Actions — install CUDA 12.6 headlessly
- name: Install CUDA Toolkit
  run: env-doctor cuda-install 12.6 --run --yes
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Installation succeeded and `nvcc --version` passed |
| `1` | An installation step failed |
| `2` | Steps ran but verification failed |
| `130` | Interrupted (Ctrl+C) |

### CI-Aware Environment Persistence

After installation, `env-doctor` automatically persists `PATH` and `LD_LIBRARY_PATH` using the right method for each CI system:

| CI System | Persistence Method |
|-----------|-------------------|
| GitHub Actions | Write to `$GITHUB_ENV` / `$GITHUB_PATH` |
| GitLab CI | Echo export commands (dotenv artifact) |
| CircleCI | Append to `$BASH_ENV` |
| Azure Pipelines | `##vso[task.setvariable]` syntax |
| Jenkins | Echo export commands |
| Generic CI (`CI=true`) | Echo export commands |
| Local Linux | Append to `~/.bashrc` or `~/.zshrc` |
| Local Windows | `setx` (PATH handled by winget) |

### GitHub Actions Full Example

```yaml
name: Install CUDA and Validate
on: [workflow_dispatch]

jobs:
  cuda-setup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install env-doctor

      - name: Install CUDA Toolkit
        run: env-doctor cuda-install --run --yes

      - name: Validate environment
        run: env-doctor check --ci
```

### Preview Before Running (`--dry-run`)

Use `--dry-run` to validate what would be executed without making any changes — useful for reviewing CI config:

```bash
env-doctor cuda-install --dry-run
```

Output:
```
[DRY RUN] [1/4] wget https://developer.download.nvidia.com/.../cuda-keyring_1.1-1_all.deb
[DRY RUN] [2/4] sudo dpkg -i cuda-keyring_1.1-1_all.deb
[DRY RUN] [3/4] sudo apt-get update
[DRY RUN] [4/4] sudo apt-get -y install cuda-toolkit-12-6
[DRY RUN] [1/1] nvcc --version

CUDA 12.6 installation completed successfully.
Verification: PASSED

Full log: /home/user/.env-doctor/install.log
```

### JSON Output from `--run`

Combine `--run --json` for machine-readable install results:

```bash
env-doctor cuda-install --run --yes --json
```

```json
{
  "success": true,
  "cuda_version": "12.6",
  "platform_key": "linux_ubuntu_22.04_x86_64",
  "steps_completed": [
    {"command": "sudo apt-get update", "phase": "install", "success": true, "return_code": 0, "duration_seconds": 8.2},
    {"command": "sudo apt-get -y install cuda-toolkit-12-6", "phase": "install", "success": true, "return_code": 0, "duration_seconds": 142.1},
    {"command": "nvcc --version", "phase": "verify", "success": true, "return_code": 0, "duration_seconds": 0.1}
  ],
  "steps_remaining": [],
  "env_vars_set": {"PATH": "/usr/local/cuda-12.6/bin:...", "LD_LIBRARY_PATH": "/usr/local/cuda-12.6/lib64"},
  "verification_passed": true,
  "error_message": null,
  "log_file": "/home/runner/.env-doctor/install.log"
}
```

## Conditional Installation

Install GPU or CPU packages based on environment:

```bash
#!/bin/bash

# Check if GPU is available
if env-doctor check --json | jq -e '.checks.driver.detected' > /dev/null; then
  echo "GPU detected, installing CUDA version"
  pip install torch torchvision
else
  echo "No GPU, installing CPU version"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
```

## Pre-commit Hook

Validate Dockerfiles before commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-dockerfile
        name: Validate Dockerfile
        entry: env-doctor dockerfile --ci
        language: system
        files: Dockerfile
        pass_filenames: false
```

## Monitoring Integration

Store results for tracking over time:

```python
import json
import subprocess
from datetime import datetime

def collect_env_metrics():
    result = subprocess.run(
        ["env-doctor", "check", "--json"],
        capture_output=True,
        text=True
    )
    data = json.loads(result.stdout)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "status": data["status"],
        "driver_version": data["checks"]["driver"].get("version"),
        "cuda_version": data["checks"]["cuda"].get("version"),
        "issues_count": data["summary"]["issues_count"]
    }

# Send to your monitoring system
metrics = collect_env_metrics()
# monitoring_client.send(metrics)
```

## See Also

- [check Command](../commands/check.md) - Full command reference
- [GitHub Actions Example](https://github.com/mitulgarg/env-doctor/tree/main/examples/github-actions)
