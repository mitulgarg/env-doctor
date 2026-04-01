# init

Generate CI/CD integration files for your project.

## Usage

```bash
env-doctor init --github-actions
```

## What It Does

Creates a `.github/workflows/env-doctor.yml` file in your project that runs `env-doctor check --ci` on every push and pull request to `main`/`master`.

The generated workflow:

- Checks out your code
- Sets up Python
- Installs env-doctor
- Runs `env-doctor check --ci` with proper exit codes

## Options

| Flag | Description |
|------|-------------|
| `--github-actions` | Generate a GitHub Actions workflow file |
| `--force` | Overwrite the file if it already exists |

## Example

```bash
$ env-doctor init --github-actions
Created .github/workflows/env-doctor.yml

Next steps:
  1. Review and edit .github/workflows/env-doctor.yml to fit your project
  2. Commit and push to activate the workflow

Common customizations:
  - Change Python version to match your project
  - Change runner to [self-hosted, gpu] for GPU validation
  - Add 'env-doctor cuda-info --json' for detailed CUDA reports
```

## Generated Workflow

```yaml
name: Validate ML Environment

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  validate-env:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install env-doctor
        run: pip install env-doctor

      - name: Validate environment
        run: env-doctor check --ci
```

After generating, customize the file to fit your project — change the Python version, switch to a GPU runner, or add extra steps like `env-doctor cuda-info --json`.

## See Also

- [CI/CD Integration Guide](../guides/ci-cd.md) - Full CI/CD setup including GitLab CI, JSON parsing, and automated CUDA installation
- [check Command](check.md) - Details on `--ci` and `--json` flags
