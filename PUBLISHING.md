# Publishing Guide for env-doctor

This guide covers how to publish env-doctor to TestPyPI and PyPI, and create GitHub releases.

## Prerequisites

1. **TestPyPI Account**: Create an account at https://test.pypi.org/account/register/
2. **TestPyPI API Token**: You mentioned you already have this. Store it securely.
3. **Build Tools**: Install required tools:
   ```bash
   pip install --upgrade build twine
   ```

## Step 1: Prepare the Release

### 1.1 Verify Version Number
Check that the version in these files matches:
- `pyproject.toml` (line 7): `version = "0.1.0"`
- `src/env_doctor/__init__.py` (line 1): `__version__ = "0.1.0"`

### 1.2 Clean Previous Builds
Remove any old build artifacts:
```bash
rm -rf dist/ build/ src/*.egg-info
```

On Windows:
```powershell
Remove-Item -Recurse -Force dist, build, src\*.egg-info -ErrorAction SilentlyContinue
```

### 1.3 Update README (Optional)
Update the README.md installation section once published:
```markdown
## Installation

```bash
# From TestPyPI (for testing)
pip install -i https://test.pypi.org/simple/ env-doctor

# From PyPI (official release - coming soon)
pip install env-doctor
```
```

## Step 2: Build the Distribution

Build both wheel and source distribution:
```bash
python -m build
```

This creates:
- `dist/env_doctor-0.1.0-py3-none-any.whl` (wheel)
- `dist/env-doctor-0.1.0.tar.gz` (source distribution)

### Verify the Build
Check that the distribution is valid:
```bash
twine check dist/*
```

Expected output:
```
Checking dist/env_doctor-0.1.0-py3-none-any.whl: PASSED
Checking dist/env-doctor-0.1.0.tar.gz: PASSED
```

## Step 3: Upload to TestPyPI

### 3.1 Upload Using API Token
```bash
twine upload --repository testpypi dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Your TestPyPI API token (starts with `pypi-...`)

Alternative (using environment variables):
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your-testpypi-api-token-here
twine upload --repository testpypi dist/*
```

On Windows (PowerShell):
```powershell
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="your-testpypi-api-token-here"
twine upload --repository testpypi dist/*
```

### 3.2 Verify Upload
Visit: https://test.pypi.org/project/env-doctor/

You should see your package with version 0.1.0.

## Step 4: Test Installation from TestPyPI

Create a new virtual environment and test installation:
```bash
# Create fresh environment
python -m venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ env-doctor

# Test the package
env-doctor check
doctor --help
```

**Note**: The `--extra-index-url` is needed because TestPyPI doesn't have your dependencies (nvidia-ml-py, click, etc.), so pip will fetch them from the regular PyPI.

## Step 5: Create GitHub Release (v0.1.0)

### 5.1 Create a Git Tag
First, ensure all changes are committed:
```bash
git status
git add .
git commit -m "Prepare for v0.1.0 release"
```

Create and push the tag:
```bash
# Create annotated tag
git tag -a v0.1.0 -m "Release version 0.1.0 - Initial public release"

# Push tag to GitHub
git push origin v0.1.0
```

### 5.2 Create GitHub Release via Web UI

1. Go to: https://github.com/mitulgarg/env-doctor/releases/new
2. Fill in:
   - **Tag**: Select `v0.1.0` (the tag you just created)
   - **Release title**: `v0.1.0 - Initial Release`
   - **Description**: Use the template below

#### Release Description Template:
```markdown
# 🎉 Initial Public Release - v0.1.0

This is the first official release of **Env-Doctor**, a CLI tool that diagnoses and fixes compatibility issues between GPU drivers, CUDA toolkits, and Python AI libraries.

## 🚀 Features

- **Environment Diagnosis**: Check compatibility between GPU Driver, CUDA Toolkit, and Python libraries
- **WSL2 GPU Support**: Detect WSL1/WSL2 environments and validate GPU forwarding
- **CUDA Analysis**: Deep inspection of CUDA installations, environment variables, and configurations
- **cuDNN Detection**: Find and validate cuDNN libraries
- **Docker Validation**: Validate Dockerfiles and docker-compose.yml for GPU configuration issues
- **Safe Install Commands**: Get the exact `pip install` command that works with your driver
- **Code Scanning**: Scan projects for deprecated imports (LangChain, Pydantic)

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install env-doctor
```

### From TestPyPI (Testing)
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ env-doctor
```

### From Source
```bash
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor
pip install -e .
```

## 🛠️ Quick Start

```bash
# Diagnose your environment
env-doctor check

# Get detailed CUDA analysis
env-doctor cuda-info

# Get safe install command for PyTorch
env-doctor install torch

# Validate your Dockerfile
env-doctor dockerfile
```

## 📋 Requirements

- Python 3.7+
- Linux (native or WSL2) or Windows
- NVIDIA GPU (for GPU-related diagnostics)

## 🐛 Known Issues

None at this time. Please report issues at https://github.com/mitulgarg/env-doctor/issues

## 📝 Full Documentation

See the [README](https://github.com/mitulgarg/env-doctor/blob/main/README.md) for complete documentation.

---

**What's Next?** We're working on expanding model compatibility checking, adding more AI frameworks, and improving WSL2 diagnostics. Stay tuned!
```

3. **Attachments** (Optional): Upload the distribution files:
   - `dist/env_doctor-0.1.0-py3-none-any.whl`
   - `dist/env-doctor-0.1.0.tar.gz`

4. Click **"Publish release"**

### 5.3 Create GitHub Release via CLI (Alternative)

If you have `gh` CLI installed:
```bash
gh release create v0.1.0 \
  --title "v0.1.0 - Initial Release" \
  --notes-file release-notes.md \
  dist/env_doctor-0.1.0-py3-none-any.whl \
  dist/env-doctor-0.1.0.tar.gz
```

## Step 6: Publish to PyPI (When Ready)

When you're ready to publish to the official PyPI (after testing on TestPyPI):

1. **Get PyPI API Token**: Create at https://pypi.org/manage/account/token/

2. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

   Credentials:
   - **Username**: `__token__`
   - **Password**: Your PyPI API token

3. **Verify**: Visit https://pypi.org/project/env-doctor/

## Common Issues and Solutions

### Issue: "File already exists"
**Cause**: You're trying to upload a version that already exists.
**Solution**: Increment the version number in `pyproject.toml` and `__init__.py`, rebuild, and upload again.

### Issue: "Invalid distribution"
**Cause**: Missing required metadata or malformed files.
**Solution**: Run `twine check dist/*` to identify issues.

### Issue: "Wheel is not PEP 517 compatible"
**Cause**: Build system issues.
**Solution**: Ensure `pyproject.toml` has correct `[build-system]` configuration.

### Issue: Dependencies not installing from TestPyPI
**Cause**: Dependencies don't exist on TestPyPI.
**Solution**: Use `--extra-index-url https://pypi.org/simple/` when installing.

## Version Management

For future releases:

1. Update version in:
   - `pyproject.toml`
   - `src/env_doctor/__init__.py`

2. Follow semantic versioning:
   - **Patch** (0.1.X): Bug fixes
   - **Minor** (0.X.0): New features, backward compatible
   - **Major** (X.0.0): Breaking changes

3. Update CHANGELOG.md (create if needed)

## Checklist for Each Release

- [ ] Update version numbers in `pyproject.toml` and `__init__.py`
- [ ] Clean old build artifacts (`rm -rf dist/ build/ *.egg-info`)
- [ ] Build distributions (`python -m build`)
- [ ] Verify build (`twine check dist/*`)
- [ ] Upload to TestPyPI (`twine upload --repository testpypi dist/*`)
- [ ] Test installation from TestPyPI in clean environment
- [ ] Create and push git tag (`git tag -a vX.Y.Z`)
- [ ] Create GitHub release with release notes
- [ ] Upload to PyPI (`twine upload dist/*`)
- [ ] Verify on PyPI
- [ ] Update README.md if needed

## Security Note

**NEVER commit API tokens to version control!**
- Store tokens securely (password manager)
- Use environment variables or `.pypirc` (excluded from git)
- For GitHub Actions, use repository secrets

## Need Help?

- **PyPI Documentation**: https://packaging.python.org/
- **TestPyPI**: https://test.pypi.org/
- **Twine Guide**: https://twine.readthedocs.io/
- **Project Issues**: https://github.com/mitulgarg/env-doctor/issues