# Contributing to env-doctor

Thank you for your interest in contributing! Accurate environment detection is a hard problem, and community input makes a real difference.

> **📋 Testing Policy**: All code changes (features and bug fixes) must include tests and pass the existing test suite. See [Testing Requirements](#testing-requirements-️) below.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork and set up a dev environment:

```bash
git clone https://github.com/<your-username>/env-doctor.git
cd env-doctor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

You can now run `env-doctor` from your terminal. Changes to the code will reflect immediately.

## Reporting Bugs

If you find a bug or misdiagnosis (e.g., env-doctor flags a working driver as incompatible), please [open an issue](https://github.com/mitulgarg/env-doctor/issues) with:

- Your OS and GPU driver version
- Output of `env-doctor check`
- What you expected vs. what happened

## Testing Requirements ⚠️

**All pull requests must include tests and pass existing tests.**

### Running Tests

After setting up your development environment, install pytest and run the test suite:

```bash
pip install pytest
python -m pytest
```

Or run specific test types:
```bash
# Run only unit tests
python -m pytest tests/unit/

# Run only integration tests
python -m pytest tests/integration/

# Run with verbose output
python -m pytest -v
```

### When to Add Tests

| Change Type | Test Required | Test Location |
|-------------|---------------|---------------|
| **New Feature** | ✅ Yes | `tests/unit/` or `tests/integration/` |
| **Bug Fix** | ✅ Yes | Add test that reproduces the bug |
| **Detector Changes** | ✅ Yes | `tests/unit/detectors/` |
| **CLI Changes** | ✅ Yes | `tests/integration/` |
| **Documentation Only** | ❌ No | N/A |

### Writing Good Tests

- **Unit tests** for individual functions/classes in `tests/unit/`
- **Integration tests** for CLI commands in `tests/integration/`
- Use descriptive test names: `test_cuda_detector_finds_nvcc_version()`
- Include edge cases and error conditions
- Mock external dependencies (don't require actual GPU for tests)

### Example Test Structure

```python
# tests/unit/detectors/test_my_detector.py
import pytest
from env_doctor.detectors.my_detector import MyDetector

def test_detector_finds_valid_installation():
    """Test that detector correctly identifies valid installation."""
    detector = MyDetector()
    result = detector.detect()
    assert result.detected is True
    assert result.version is not None

def test_detector_handles_missing_installation():
    """Test that detector gracefully handles missing installation."""
    # Test implementation here
    pass
```

## Pull Request Guidelines

Before submitting a PR:

1. **Add tests** for bug fixes and new features (see [Testing Requirements](#testing-requirements-️))
2. **Run tests locally** and make sure they pass: `python -m pytest`
3. **Write a clear description** of what changed and why
4. Keep changes focused—one fix or feature per PR

When opening your PR, briefly mention:
- What the change does
- What tests you added
- Any breaking changes or migration notes (if applicable)

PRs without tests (for code changes) will be asked to add them before merging.

## Code Style

- Keep it simple and readable
- Minimal dependencies
- Follow existing patterns in the codebase

## Questions?

Open an issue or start a discussion. We're happy to help.