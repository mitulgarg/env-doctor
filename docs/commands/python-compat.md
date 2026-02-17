# python-compat

Check Python version compatibility with installed AI libraries and detect dependency cascades.

## Usage

```bash
env-doctor python-compat [--json]
```

## Description

The `python-compat` command checks if your current Python version is compatible with installed AI libraries like PyTorch, TensorFlow, JAX, NumPy, SciPy, and others. It identifies:

- **Version conflicts**: When your Python version is outside a library's supported range
- **Dependency cascades**: When one library's constraint forces version limits on downstream packages

## Options

| Option | Description |
|--------|-------------|
| `--json` | Output results in JSON format for programmatic consumption |

## Output

### Human-readable format (default)

```
🐍  PYTHON VERSION COMPATIBILITY CHECK
============================================================
Python Version: 3.13 (3.13.0)
Libraries Checked: 2

❌  2 compatibility issue(s) found:

    tensorflow:
      tensorflow supports Python <=3.12, but you have Python 3.13
      Note: TensorFlow 2.15+ requires Python 3.9-3.12. Python 3.13 not yet supported.

    torch:
      torch supports Python <=3.12, but you have Python 3.13
      Note: PyTorch 2.x supports Python 3.9-3.12. Python 3.13 support experimental.

⚠️   Dependency Cascades:
    tensorflow [high]: TensorFlow's Python ceiling propagates to keras and tensorboard
      Affected: keras, tensorboard, tensorflow-estimator
    torch [high]: PyTorch's Python version constraint affects all torch ecosystem packages
      Affected: torchvision, torchaudio, triton

💡  Consider using Python 3.12 or lower for full compatibility

💡  Cascade: tensorflow constraint also affects: keras, tensorboard, tensorflow-estimator

💡  Cascade: torch constraint also affects: torchvision, torchaudio, triton

============================================================
```

### JSON format (`--json`)

```json
{
  "component": "python_compat",
  "status": "error",
  "detected": false,
  "version": "3.13",
  "path": null,
  "metadata": {
    "python_full_version": "3.13.0",
    "conflicts": [
      {
        "library": "tensorflow",
        "min_version": "3.9",
        "max_version": "3.12",
        "notes": "TensorFlow 2.15+ requires Python 3.9-3.12. Python 3.13 not yet supported.",
        "type": "above_maximum",
        "message": "tensorflow supports Python <=3.12, but you have Python 3.13"
      },
      {
        "library": "torch",
        "min_version": "3.9",
        "max_version": "3.12",
        "notes": "PyTorch 2.x supports Python 3.9-3.12. Python 3.13 support experimental.",
        "type": "above_maximum",
        "message": "torch supports Python <=3.12, but you have Python 3.13"
      }
    ],
    "cascades": [
      {
        "root_library": "tensorflow",
        "affected_dependencies": ["keras", "tensorboard", "tensorflow-estimator"],
        "severity": "high",
        "description": "TensorFlow's Python ceiling propagates to keras and tensorboard"
      },
      {
        "root_library": "torch",
        "affected_dependencies": ["torchvision", "torchaudio", "triton"],
        "severity": "high",
        "description": "PyTorch's Python version constraint affects all torch ecosystem packages"
      }
    ],
    "constraints_checked": 2
  },
  "issues": [
    "tensorflow supports Python <=3.12, but you have Python 3.13",
    "torch supports Python <=3.12, but you have Python 3.13"
  ],
  "recommendations": [
    "Consider using Python 3.12 or lower for full compatibility",
    "Cascade: tensorflow constraint also affects: keras, tensorboard, tensorflow-estimator",
    "Cascade: torch constraint also affects: torchvision, torchaudio, triton"
  ]
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All installed libraries are compatible with current Python version |
| `1` | Compatibility issues detected (conflicts or warnings) |

## Use Cases

### 1. Before upgrading Python

Check if your installed libraries support the new Python version:

```bash
# On Python 3.12
env-doctor python-compat

# Check what happens if you upgrade
pyenv install 3.13
pyenv shell 3.13
env-doctor python-compat
```

### 2. After fresh Python install

Verify that your planned AI stack will work:

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install torch tensorflow
env-doctor python-compat
```

### 3. CI/CD pipelines

Ensure Python version compatibility in automated workflows:

```yaml
- name: Check Python compatibility
  run: |
    pip install env-doctor
    env-doctor python-compat --json
```

### 4. Debugging import errors

When libraries fail to import, check if Python version is the issue:

```bash
# If you see: "ImportError: DLL load failed" or "ModuleNotFoundError"
env-doctor python-compat
```

## What Libraries Are Checked?

The detector currently tracks compatibility for:

- **Core AI frameworks**: PyTorch, TensorFlow, JAX
- **Scientific computing**: NumPy, SciPy
- **ML tools**: ONNX Runtime, Transformers
- **GPU acceleration**: Triton

The compatibility data is maintained in `python_compatibility.json` and updated regularly.

## Understanding Dependency Cascades

A **dependency cascade** occurs when a root library's Python version constraint propagates to all its dependencies. For example:

```
tensorflow (requires Python ≤3.12)
  ├─ keras (forced to ≤3.12)
  ├─ tensorboard (forced to ≤3.12)
  └─ tensorflow-estimator (forced to ≤3.12)
```

Even if `keras` itself supports Python 3.13, it can't be used with TensorFlow on Python 3.13 because TensorFlow is the root constraint.

Cascades are marked with severity:
- **high**: Affects many popular packages
- **medium**: Affects a moderate number of packages
- **low**: Limited impact

## Integration with `check` command

The `python-compat` detector is automatically included in the main `check` command:

```bash
env-doctor check
```

Output will include a Python compatibility section:
```
✅  Python 3.12: Compatible with all 3 checked libraries
```

## Related Commands

- [`check`](check.md) - Full environment diagnosis (includes Python compatibility)
- [`install`](install.md) - Get safe install commands for libraries
- [`debug`](debug.md) - Detailed detector output including Python compatibility

## Troubleshooting

### No libraries checked (constraints_checked: 0)

This means none of the tracked libraries are installed. Install at least one AI library:

```bash
pip install torch
env-doctor python-compat
```

### False positives

The compatibility data is based on library documentation and may lag behind actual support. If you believe a conflict is incorrect, please [open an issue](https://github.com/mitulgarg/env-doctor/issues) with:
- Python version
- Library version
- Evidence of compatibility (e.g., successful import)

## See Also

- [Getting Started](../getting-started.md)
- [MCP Integration](../guides/mcp-integration.md) - Use `python_compat_check` tool in AI assistants
