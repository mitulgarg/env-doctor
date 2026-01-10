# Adding Models to env-doctor

This guide explains how to add new AI models to env-doctor's model compatibility database.

## Quick Start

Models are stored in `src/env_doctor/data/model_requirements.json`. To add a model, you only need its **parameter count** — everything else is optional!

## Minimum Required Data

```json
{
  "your-model-name": {
    "params_b": 7.0,
    "category": "llm",
    "family": "your-model-family"
  }
}
```

- **model-name** (key): Use lowercase with hyphens (e.g., `llama-3-8b`)
- **params_b**: Model size in billions of parameters (e.g., 7.0 for 7 billion)
- **category**: One of: `llm`, `diffusion`, `audio`, `language`
- **family**: Model family (e.g., `llama-3`, `mistral`, `stable-diffusion`)

## Full Entry Example

For maximum accuracy, include measured VRAM values:

```json
{
  "llama-3-8b": {
    "params_b": 8.0,
    "category": "llm",
    "family": "llama-3",
    "hf_id": "meta-llama/Meta-Llama-3-8B",
    "vram": {
      "fp16": 19200,
      "int4": 4800
    },
    "notes": "Instruction-tuned variant, best for instruction following"
  }
}
```

Optional fields:
- **hf_id**: HuggingFace model ID (for reference links)
- **vram**: Measured VRAM in MB for specific precisions (see below)
- **notes**: Implementation details or usage notes

## Finding Parameter Counts

### Option 1: HuggingFace Model Card (Recommended)

1. Visit the model on HuggingFace: `https://huggingface.co/[author]/[model-name]`
2. Look for the parameter count in:
   - Model description/tags (often shows "70B" or "Parameters: 70 billion")
   - Model card text
   - Technical specs section

**Example search:**
```
https://huggingface.co/meta-llama/Meta-Llama-3-70B
```

### Option 2: Google Search

```
"[model name] parameters billion" OR "[model name] size"
```

Example: `"Llama 3 70B parameters"` → Result: "70 billion"

### Option 3: GitHub Repository

Check the model's official GitHub repository:
- Look in README.md
- Check model_config.json files
- Search for "params" or "parameters"

### Option 4: Official Documentation

Most model providers publish parameter counts in:
- Official model cards
- Research papers (usually in abstract or table)
- Blog posts announcing the model

## Adding Measured VRAM Values (Optional, Advanced)

If you have access to the model and a GPU, you can measure actual VRAM usage for more accuracy.

### How to Measure VRAM

1. Load model in specific precision:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or torch.bfloat16, torch.float32
    device_map="auto"
)
```

2. Check VRAM usage:

```bash
nvidia-smi
```

Look for the GPU memory usage of your process.

3. Record the value:

```json
{
  "vram": {
    "fp16": 19200,  # in MB
    "int4": 4800
  }
}
```

### VRAM Values Format

- **Key**: Precision (fp32, fp16, bf16, int8, int4, fp8)
- **Value**: VRAM in **MB** (megabytes), not GB

Example conversions:
- 19.2 GB = 19,200 MB
- 4.8 GB = 4,800 MB
- 140 GB = 140,000 MB

### Which Precisions to Measure?

Start with the most common:
1. **fp16** - Standard inference precision
2. **int4** - Quantized, most popular for memory efficiency
3. **int8** - 8-bit quantized
4. **bf16** - Brain float (if model supports it)

## Model Categories

Choose the most appropriate category:

| Category | Examples | Use When |
|----------|----------|----------|
| `llm` | Llama, Mistral, Qwen, Mixtral | Language model for text generation |
| `diffusion` | Stable Diffusion, FLUX, Kandinsky | Image generation or image manipulation |
| `audio` | Whisper, Musicgen | Speech recognition, generation, or processing |
| `language` | BERT, T5, RoBERTa | Text encoding, classification, or small language tasks |

## Model Naming Conventions

**Do:**
- Use lowercase: `llama-3-8b` ✅
- Use hyphens: `stable-diffusion-xl` ✅
- Include size: `mixtral-8x7b` ✅
- Be descriptive: `bert-base-uncased` ✅

**Don't:**
- Use spaces: `llama 3 8b` ❌
- Use underscores for separation: `llama_3_8b` ❌
- Abbreviate: `sd-xl` instead of `stable-diffusion-xl` ❌
- Include version number: `v1`, `v2.0` ❌ (unless it's part of official name)

## Adding Aliases

Help users find models with alternative names:

```json
{
  "aliases": {
    "llama3-8b": "llama-3-8b",
    "sdxl": "stable-diffusion-xl",
    "mistral-7b-v01": "mistral-7b"
  }
}
```

Aliases are case-insensitive and automatically resolved.

## Database Schema Validation

All submitted models must pass validation:

```bash
# Run validation tests
pytest tests/unit/test_vram_calculator.py::TestVRAMCalculatorDatabaseIntegrity -v
```

**Automatic checks:**
- ✅ All models have `params_b` > 0
- ✅ All `category` values are valid
- ✅ All aliases point to existing models
- ✅ VRAM values are reasonable (0 < x < 1,000,000 MB)
- ✅ Parameter counts are within realistic ranges

## Submitting Your Changes

### Step 1: Create Your Branch

```bash
git checkout -b feature/add-models
```

### Step 2: Edit the Database

Add your models to `src/env_doctor/data/model_requirements.json`:

```bash
nano src/env_doctor/data/model_requirements.json
# or use your preferred editor
```

### Step 3: Test Your Changes

```bash
# Run database validation tests
pytest tests/unit/test_vram_calculator.py::TestVRAMCalculatorDatabaseIntegrity -v

# Test the model works via CLI
env-doctor model your-model-name
env-doctor model --list | grep your-model-name
```

### Step 4: Commit and Push

```bash
git add src/env_doctor/data/model_requirements.json
git commit -m "feat: add [model-name] and variants to model database"
git push origin feature/add-models
```

### Step 5: Create a Pull Request

Include in your PR description:
- Which models were added
- Parameter counts and sources
- Any measured VRAM values included
- Models tested (if you have GPU access)

**Example PR title:**
```
feat: add Llama-3 variants (8B, 70B, 405B) to model database
```

## Common Questions

### Q: What if I don't know exact parameter count?

**A:** Use a reasonable estimate or measurement. For example:
- Look at model size comparisons to known models
- Check papers or documentation
- Estimate from download size (rough: GB file size × 2 ≈ parameters)

### Q: Can I add multiple models in one PR?

**A:** Yes! Grouping related models (e.g., all Llama-3 variants) is encouraged.

### Q: How do I find VRAM requirements if I don't have a GPU?

**A:** The formula-based calculation is good enough for initial release. Many users measure and contribute VRAM values later.

### Q: What if a model has multiple variants (quantized, instruct, chat)?

**A:** Add the base model. If variants have significantly different parameter counts, add them separately:

```json
{
  "llama-3-8b": { "params_b": 8.0, ... },
  "llama-3-8b-instruct": { "params_b": 8.0, ... }
}
```

### Q: How do I find the HuggingFace model ID?

**A:** It's in the URL on HuggingFace:

```
https://huggingface.co/meta-llama/Meta-Llama-3-8B
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  <- This is the hf_id
```

### Q: What's the correct value for "family"?

**A:** Use the model series name, lowercase with hyphens:
- `llama-3` (not `llama3` or `llama 3`)
- `stable-diffusion` (not `sd` or `stable-diff`)
- `mistral` (single models don't need numbers)

## Examples

### Adding a Single Model

```json
{
  "qwen-7b": {
    "params_b": 7.0,
    "category": "llm",
    "family": "qwen",
    "hf_id": "Qwen/Qwen-7B"
  }
}
```

### Adding a Model Family with Variants

```json
{
  "mixtral-8x7b": {
    "params_b": 46.7,
    "category": "llm",
    "family": "mixtral",
    "hf_id": "mistralai/Mixtral-8x7B-v0.1",
    "notes": "Mixture of Experts: 46.7B total, 12.9B active"
  },
  "mixtral-8x22b": {
    "params_b": 176.0,
    "category": "llm",
    "family": "mixtral",
    "hf_id": "mistral-community/Mixtral-8x22B-v0.1",
    "vram": {
      "fp16": 263000
    }
  }
}
```

### Adding with Measured VRAM

```json
{
  "stable-diffusion-xl": {
    "params_b": 3.5,
    "category": "diffusion",
    "family": "stable-diffusion",
    "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "vram": {
      "fp16": 8000
    },
    "notes": "SDXL base model, improved quality over v1.5"
  }
}
```

## Need Help?

- 📚 Check existing models in `src/env_doctor/data/model_requirements.json`
- 🐛 Open an issue if you find errors
- 💬 Discuss in GitHub discussions for questions
- 🔗 Reference HuggingFace model cards for official specs

## Contribution Recognition

Thank you for contributing! Your additions help the community:
- Make env-doctor more useful for more models
- Build an accurate VRAM database
- Enable better recommendations for all users

Contributors are recognized in:
- GitHub commit history
- Pull request discussions
- Model database comments (if notable measurements included)

## Future Enhancements

Models you add today enable future features like:
- Fine-tuning VRAM requirements
- Batch size optimization recommendations
- Context length impact on VRAM
- Multi-GPU sharding strategies
