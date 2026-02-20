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
