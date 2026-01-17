# docker-compose

Validate docker-compose.yml files for proper GPU device configuration.

## Usage

```bash
env-doctor docker-compose [PATH]
```

If no path is provided, looks for `docker-compose.yml` or `docker-compose.yaml` in the current directory.

## What It Validates

### GPU Device Configuration

- Checks for `deploy.resources.reservations.devices`
- Validates `driver: nvidia` is specified
- Ensures `capabilities: [gpu]` is set

### Deprecated Syntax

- Flags old `runtime: nvidia` approach
- Provides migration path to new syntax

### Multi-Service Configuration

- Warns about GPU resource sharing between services
- Validates each service with GPU requirements

### Host Requirements

- Checks for nvidia-container-toolkit
- Validates Docker GPU support

## Example Output

```
🐳  DOCKER COMPOSE VALIDATION: docker-compose.yml

❌  ERRORS (1):
------------------------------------------------------------

Service 'ml-training':
  Issue: Missing GPU device configuration
  Fix:   Add GPU device configuration under deploy.resources.reservations.devices

  Suggested fix:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

⚠️   WARNINGS (1):
------------------------------------------------------------

Service 'legacy-app':
  Issue: Deprecated 'runtime: nvidia' syntax
  Fix:   Use the new 'deploy.resources.reservations.devices' syntax instead

SUMMARY:
  ❌ Errors:   1
  ⚠️  Warnings: 1
  ℹ️  Info:     0
```

## Correct GPU Configuration

### Modern Syntax (Compose v2.4+)

```yaml
services:
  ml-training:
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # or specific number: count: 1
              capabilities: [gpu]
```

### Device Selection

```yaml
# Use all GPUs
count: all

# Use specific number of GPUs
count: 2

# Use specific GPU by ID
device_ids: ['0', '2']
```

### Deprecated Syntax

```yaml
# ❌ Old syntax (still works but deprecated)
services:
  app:
    runtime: nvidia

# ✅ New syntax
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```

## Common Issues

### Missing nvidia-container-toolkit

```
❌ nvidia-container-toolkit not detected on host

💡 Installation:
  # Ubuntu/Debian
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt update && sudo apt install nvidia-container-toolkit
  sudo systemctl restart docker
```

### GPU Sharing Warning

```
⚠️  Multiple services reserving GPU resources

  Services: ml-training, inference-api

💡 Note:
  GPU memory is shared between containers.
  Ensure total VRAM requirements don't exceed available memory.
```

## See Also

- [dockerfile](dockerfile.md) - Validate Dockerfile
