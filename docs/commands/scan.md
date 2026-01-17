# scan

Scan your project for deprecated AI library imports and suggest fixes.

## Usage

```bash
env-doctor scan [PATH]
```

If no path is provided, scans the current directory.

## What It Detects

### LangChain Deprecations

- Old import paths from `langchain`
- Migration to `langchain-core`, `langchain-community`, etc.

### Pydantic V1 → V2

- Deprecated validator syntax
- Old import paths

### Other Library Changes

- TensorFlow 1.x patterns
- PyTorch deprecated APIs

## Example Output

```
🔍 SCANNING PROJECT: ./my_project

📁 Files scanned: 47
📝 Issues found: 3

------------------------------------------------------------

src/chains/qa.py:12
  ❌ Deprecated import: from langchain.chat_models import ChatOpenAI
  ✅ Replace with: from langchain_openai import ChatOpenAI

src/chains/qa.py:15
  ❌ Deprecated import: from langchain.embeddings import OpenAIEmbeddings
  ✅ Replace with: from langchain_openai import OpenAIEmbeddings

src/models/user.py:8
  ⚠️  Pydantic V1 syntax: @validator
  ✅ Replace with: @field_validator (Pydantic V2)

SUMMARY:
  ❌ Errors:   2
  ⚠️  Warnings: 1
```

## Scan Patterns

### LangChain

| Old Import | New Import |
|------------|------------|
| `langchain.chat_models` | `langchain_openai` or `langchain_anthropic` |
| `langchain.embeddings` | `langchain_openai` |
| `langchain.vectorstores` | `langchain_community.vectorstores` |
| `langchain.document_loaders` | `langchain_community.document_loaders` |

### Pydantic

| V1 Syntax | V2 Syntax |
|-----------|-----------|
| `@validator` | `@field_validator` |
| `@root_validator` | `@model_validator` |
| `class Config:` | `model_config = ConfigDict(...)` |

## Excluding Files

The scan automatically excludes:

- `.git/`
- `__pycache__/`
- `node_modules/`
- `.venv/`, `venv/`
- `*.pyc`, `*.pyo`

---

## Advanced Options

### JSON Output

```bash
env-doctor scan --json
```

## See Also

- [check](check.md) - Environment diagnosis