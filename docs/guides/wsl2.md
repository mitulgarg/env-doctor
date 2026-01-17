# WSL2 GPU Support

Env-Doctor provides comprehensive WSL2 environment detection and GPU forwarding validation.

## Environment Detection

Env-Doctor automatically detects your environment:

| Environment | GPU Support |
|-------------|-------------|
| Native Linux | Full CUDA support |
| WSL1 | No GPU support (CUDA not available) |
| WSL2 | GPU forwarding from Windows host |

## How WSL2 GPU Works

In WSL2, GPU access is **forwarded from the Windows host**:

```
Windows Host (NVIDIA Driver) → WSL2 (CUDA libraries) → Your Application
```

Key points:

- NVIDIA driver is installed on **Windows only**
- WSL2 uses special libraries in `/usr/lib/wsl/lib/`
- Do NOT install NVIDIA drivers inside WSL2

## WSL2 Validation

Run `env-doctor check` to validate your WSL2 GPU setup:

```bash
env-doctor check
```

**Healthy WSL2 Output:**

```
🩺 ENV-DOCTOR DIAGNOSIS
============================================================

🖥️  Environment: WSL2 (GPU forwarding enabled)

🎮 GPU Driver
   ✅ NVIDIA Driver: 535.146.02 (via WSL forwarding)
   └─ Max CUDA: 12.2

✅ GPU forwarding is working correctly!
```

## Common Issues

### Driver Installed Inside WSL

**Problem:**

```
❌ NVIDIA driver installed inside WSL. This breaks GPU forwarding.
```

**Cause:** You installed `nvidia-driver-*` packages inside WSL2.

**Fix:**

```bash
sudo apt remove --purge nvidia-* libnvidia-*
sudo apt autoremove
```

Then restart WSL:

```powershell
# In PowerShell (Windows)
wsl --shutdown
```

### Missing CUDA Libraries

**Problem:**

```
❌ Missing /usr/lib/wsl/lib/libcuda.so
```

**Cause:** Windows NVIDIA driver not properly installed or outdated.

**Fix:**

1. Update Windows NVIDIA driver to 470.76 or newer
2. Download from [nvidia.com/drivers](https://www.nvidia.com/drivers)
3. Restart Windows after installation

### nvidia-smi Not Working

**Problem:**

```
❌ nvidia-smi command failed
```

**Possible causes:**

1. **Windows driver too old** - Update to 470.76+
2. **WSL version outdated** - Run `wsl --update`
3. **Hyper-V not enabled** - Required for WSL2 GPU

**Fix:**

```powershell
# Update WSL
wsl --update

# Verify WSL2 version
wsl --version
```

### Running in WSL1

**Problem:**

```
⚠️  WSL1 detected. CUDA is not supported in WSL1.
```

**Fix:** Convert to WSL2:

```powershell
# List distributions
wsl --list --verbose

# Convert to WSL2
wsl --set-version <distro-name> 2
```

## Requirements

### Windows Requirements

- Windows 10 version 21H2 or newer (or Windows 11)
- NVIDIA driver 470.76 or newer
- WSL2 (not WSL1)

### WSL2 Requirements

- **Do NOT** install NVIDIA drivers
- CUDA toolkit installation is optional (most libraries bundle it)
- cuDNN can be installed if needed

## Recommended Setup

### Minimal Setup (Most Users)

1. Install NVIDIA driver on Windows
2. Install WSL2 with Ubuntu
3. Install Python packages normally

```bash
# In WSL2
pip install torch  # Works out of the box
```

### Development Setup (For Compilation)

If you need to compile CUDA extensions:

```bash
# Install CUDA toolkit (not driver!)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1

# Set environment
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
```

## Troubleshooting

### Check Windows Driver

In PowerShell:

```powershell
nvidia-smi
```

Should show driver version 470.76+.

### Check WSL2 GPU Access

In WSL2:

```bash
nvidia-smi
```

Should show same driver as Windows.

### Check Library Paths

```bash
ls -la /usr/lib/wsl/lib/
```

Should contain `libcuda.so`, `libnvidia-ml.so`, etc.

### Full Debug Output

```bash
env-doctor debug
```

Shows detailed WSL2 detection metadata.

## See Also

- [check Command](../commands/check.md)
- [cuda-info Command](../commands/cuda-info.md)
