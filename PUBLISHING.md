# Publishing Guide for env-doctor

This guide covers how to publish new releases of env-doctor to PyPI using automated GitHub Actions.

## Current Status

- **Latest Version**: v0.1.0
- **PyPI Package**: https://pypi.org/project/env-doctor/
- **Installation**: `pip install env-doctor`
- **GitHub Releases**: https://github.com/mitulgarg/env-doctor/releases
- **Documentation**: https://mitulgarg.github.io/env-doctor/

## Release Automation

Starting with v0.1.1, all PyPI releases are **fully automated** via GitHub Actions:

1. Push a git tag (e.g., `v0.1.1`)
2. GitHub Actions automatically:
   - Runs tests
   - Builds the package
   - Waits for manual approval
   - Publishes to PyPI

**No API tokens or manual uploads needed** - uses PyPI Trusted Publishing (OIDC).

## How to Create a Release

### Step 1: Prepare the Release

#### 1.1 Update Version Number

Update the version in **both** files:
- `pyproject.toml` (line 7): `version = "0.1.1"`
- `src/env_doctor/__init__.py`: `__version__ = "0.1.1"`

#### 1.2 Update Documentation (if needed)

- Update README.md with new features/changes
- Update docs/ if significant changes
- Update CHANGELOG.md (create if it doesn't exist)

#### 1.3 Commit Changes

```bash
git add pyproject.toml src/env_doctor/__init__.py
git commit -m "chore: bump version to v0.1.1"
git push origin main
```

### Step 2: Create and Push Git Tag

```bash
# Create annotated tag
git tag -a v0.1.1 -m "Release v0.1.1: CI/CD automation and documentation improvements"

# Push tag to GitHub (this triggers the workflow)
git push origin v0.1.1
```

### Step 3: Monitor GitHub Actions

1. Go to: https://github.com/mitulgarg/env-doctor/actions
2. You'll see the "Publish to PyPI" workflow running
3. The workflow will:
   - ✅ Run tests
   - ✅ Build distribution packages
   - ⏸️ Wait for your approval

### Step 4: Approve the Release

1. In the Actions tab, click on the running workflow
2. Click the "Review deployments" button
3. Check the `pypi` environment
4. Click "Approve and deploy"

The package will be published to PyPI within 1-2 minutes.

### Step 5: Create GitHub Release (Optional)

Create a GitHub release with release notes:

```bash
gh release create v0.1.1 \
  --title "v0.1.1 - CI/CD Automation" \
  --notes "### Changes
- Added automated PyPI publishing via GitHub Actions
- Updated documentation and README
- Improved dependency version constraints in pyproject.toml
- Updated to Python 3.10+ minimum requirement

**Install**: \`pip install env-doctor\`"
```

Or use the GitHub web UI: https://github.com/mitulgarg/env-doctor/releases/new

## Version Numbering (Semantic Versioning)

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **Patch** (0.1.X): Bug fixes, documentation, internal improvements
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes

### Examples:
- v0.1.1: CI/CD automation, README updates (PATCH)
- v0.2.0: MCP server integration (MINOR - new feature)
- v1.0.0: Stable API, production ready (MAJOR)

## Rollback a Release

If you need to rollback or fix a bad release:

1. You **cannot delete** PyPI releases
2. Instead, immediately release a patch version:
   ```bash
   # If v0.1.1 is broken, release v0.1.2 with fixes
   git tag -a v0.1.2 -m "Release v0.1.2: Fix critical bug from v0.1.1"
   git push origin v0.1.2
   ```

## Troubleshooting

### Workflow fails at "Run Tests"
- Fix the failing tests
- Push fixes to main
- Delete and recreate the tag:
  ```bash
  git tag -d v0.1.1
  git push origin :refs/tags/v0.1.1
  git tag -a v0.1.1 -m "Release v0.1.1"
  git push origin v0.1.1
  ```

### Workflow fails at "Publish to PyPI"
- Check that PyPI Trusted Publishing is configured
- Verify the environment name is `pypi` (case-sensitive)
- Check workflow permissions in `.github/workflows/publish-pypi.yml`

### "File already exists" error on PyPI
- You're trying to upload a version that already exists
- Bump to the next version number and try again
- PyPI does **not** allow overwriting existing versions

## Prerequisites for Maintainers

### PyPI Trusted Publishing Setup

Already configured for this repository. If you need to reconfigure:

1. Go to https://pypi.org/manage/project/env-doctor/settings/publishing/
2. Add publisher:
   - **PyPI Project Name**: `env-doctor`
   - **Owner**: `mitulgarg`
   - **Repository name**: `env-doctor`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: `pypi`

### GitHub Environment Protection

To enable manual approval before publishing:

1. Go to: https://github.com/mitulgarg/env-doctor/settings/environments
2. Click "pypi" environment
3. Check "Required reviewers"
4. Add yourself as a reviewer

## Release Checklist

Before tagging a release:

- [ ] Version updated in `pyproject.toml` and `__init__.py`
- [ ] All tests pass (`pytest`)
- [ ] README.md updated (if needed)
- [ ] Documentation updated (if needed)
- [ ] Changes committed and pushed to main
- [ ] Ready to create git tag

After tagging:

- [ ] GitHub Actions workflow triggered
- [ ] Tests passed in CI
- [ ] Build succeeded
- [ ] Approved deployment
- [ ] Package visible on PyPI
- [ ] GitHub release created with release notes
- [ ] Verified installation: `pip install env-doctor`

## Testing Locally Before Release

Test the package build locally before pushing a tag:

```bash
# Clean old builds
rm -rf dist/ build/ src/*.egg-info

# Build package
python -m build

# Check distribution
pip install twine
twine check dist/*

# Test install in clean environment
python -m venv test-env
source test-env/bin/activate  # Windows: test-env\Scripts\activate
pip install dist/env_doctor-0.1.1-py3-none-any.whl
env-doctor check
deactivate
rm -rf test-env
```

## Security

- **No API tokens in repository**: Uses OIDC Trusted Publishing
- **Manual approval required**: Prevents accidental releases
- **Signed commits recommended**: Use GPG signing for releases

## Need Help?

- **GitHub Actions Issues**: https://github.com/mitulgarg/env-doctor/actions
- **PyPI Package**: https://pypi.org/project/env-doctor/
- **Report Issues**: https://github.com/mitulgarg/env-doctor/issues
- **PyPI Documentation**: https://packaging.python.org/