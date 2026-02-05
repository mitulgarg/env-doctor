# model

Check if an AI model fits on your GPU before downloading.

## Usage

```bash
env-doctor model <model-name>
```

## Options

| Option | Description |
|--------|-------------|
| `--list` | List all available models in local database |
| `--precision <type>` | Check specific precision (fp32, fp16, bf16, int8, int4, fp8) |

## Example

```bash
env-doctor model llama-3-8b
```

**Output:**

```
🤖  Checking: LLAMA-3-8B
    Parameters: 8.0B
    HuggingFace: meta-llama/Meta-Llama-3-8B

🖥️   Your Hardware:
    RTX 3090 (24GB VRAM)

💾  VRAM Requirements & Compatibility

  ✅  FP16: 19.2GB (measured) - 4.8GB free
  ✅  INT4:  4.8GB (estimated) - 19.2GB free

✅  This model WILL FIT on your GPU!

💡  Recommendations:
1. Use fp16 for best quality on your GPU
```

## Listing Available Models

```bash
env-doctor model --list
```

**Output:**

```
📋 Available Models

LLMs:
  llama-3-8b, llama-3-70b, llama-3-405b
  mistral-7b, mixtral-8x7b
  qwen-7b, qwen-14b, qwen-72b

Diffusion:
  stable-diffusion-1.5, stable-diffusion-xl, stable-diffusion-3
  flux-schnell, flux-dev

Audio:
  whisper-tiny, whisper-base, whisper-small
  whisper-medium, whisper-large, whisper-large-v3

Language:
  bert-base, bert-large
  t5-small, t5-base, t5-large
```

## Checking Specific Precision

```bash
env-doctor model stable-diffusion-xl --precision int4
```

**Output:**

```
🤖  Checking: STABLE-DIFFUSION-XL
    Parameters: 6.6B

🖥️   Your Hardware:
    RTX 3060 (12GB VRAM)

💾  INT4 Requirements:
    ~2.5GB VRAM (estimated)

✅  This model WILL FIT at INT4 precision!
```

## Model Database

env-doctor includes a curated local database of **75+ popular models** with measured VRAM usage, plus access to **thousands of models** via HuggingFace Hub API.

### Supported Model Categories

The local database includes models across multiple categories:

- **LLMs**: Llama 3 (8B-405B), Mistral, Mixtral, Qwen, Gemma, Phi, CodeLlama
- **Diffusion**: Stable Diffusion (1.5, XL, 3), Flux (Schnell, Dev), Pixart
- **Audio**: Whisper (all sizes), Bark
- **Vision**: CLIP, SAM, DINOv2
- **Language**: BERT, T5, RoBERTa, DistilBERT

### Example Models from Local Database

| Category | Example Models | Typical VRAM (FP16) |
|----------|----------------|---------------------|
| Small LLMs | Llama-3-8B, Mistral-7B, Gemma-7B | 14-19GB |
| Large LLMs | Llama-3-70B, Mixtral-8x7B | 93-140GB |
| Diffusion | SD 1.5, SD XL, Flux Schnell | 4-12GB |
| Audio | Whisper Tiny/Base/Large | 1-10GB |
| Vision | CLIP, SAM, BERT | 0.5-2GB |

!!! info "Beyond the Local Database"
    Can't find your model? No problem! Any public HuggingFace model can be checked automatically. See [HuggingFace API Integration](#huggingface-api-integration) below.

### Model Aliases

Common aliases are supported for quick access:

- `sdxl` → `stable-diffusion-xl`
- `sd15` → `stable-diffusion-1.5`
- `llama3` → `llama-3-8b`
- `gemma` → `gemma-7b`
- `phi2` → `phi-2`
- `codellama` → `codellama-7b`
- `clip` → `clip-vit-base`
- `sam` → `sam-vit-base`

## HuggingFace API Integration

!!! tip "New Feature"
    Models not in the local database are automatically fetched from HuggingFace Hub!

### 3-Tier Fallback System

When you query a model, env-doctor uses a smart 3-tier lookup:

```
Tier 1: Local Database (75+ models) → Fastest, measured VRAM values
    ↓ (if not found)
Tier 2: HF Cache → Previously fetched models, no network call
    ↓ (if not found)
Tier 3: HuggingFace Hub API → Dynamic fetch, then cached
```

### Checking Any HuggingFace Model

You can check **any public model** from HuggingFace Hub:

```bash
# Using HuggingFace model ID
env-doctor model bert-base-uncased
env-doctor model sentence-transformers/all-MiniLM-L6-v2
env-doctor model distilbert-base-uncased
```

**Output for HuggingFace-fetched model:**

```
🤖  Checking: BERT-BASE-UNCASED
    (Fetched from HuggingFace API - cached for future use)
    Parameters: 0.11B
    HuggingFace: bert-base-uncased

🖥️   Your Hardware:
    RTX 3090 (24GB VRAM)

💾  VRAM Requirements & Compatibility
  ✅  FP16:  264 MB - Fits easily!

💡  Recommendations:
1. Use fp16 for best quality on your GPU
```

### Automatic Caching

Once fetched, models are cached in the local database for instant lookup on future queries - no network calls needed!

```bash
# First call: fetches from HuggingFace (2-3 seconds)
env-doctor model sentence-transformers/all-MiniLM-L6-v2

# Second call: uses cache (instant)
env-doctor model sentence-transformers/all-MiniLM-L6-v2
```

### Limitations

!!! warning "Gated Models"
    HuggingFace models that require authentication (signup/access request) cannot be fetched automatically. Use models from the local database or public HuggingFace models.

## When a Model Won't Fit

```bash
env-doctor model llama-3-70b
```

```
🤖  Checking: LLAMA-3-70B
    Parameters: 70B

🖥️   Your Hardware:
    RTX 3090 (24GB VRAM)

💾  VRAM Requirements:
    FP16: ~140GB
    INT4: ~35GB

❌  This model will NOT fit on your GPU

💡  Recommendations:
1. Try llama-3-8b (same family, fits in 24GB)
2. Use INT4 quantization with 2x RTX 3090
3. Consider cloud GPU (A100 80GB)
```

## How VRAM Is Calculated

For models with **measured data**, we use real-world VRAM usage from testing.

For other models, we estimate using:

| Precision | Formula |
|-----------|---------|
| FP32 | params × 4 bytes |
| FP16 | params × 2 bytes |
| INT8 | params × 1 byte + overhead |
| INT4 | params × 0.5 bytes + overhead |

!!! note "Overhead"
    Actual VRAM usage includes KV cache, activations, and framework overhead.
    Our estimates include a ~20% buffer for this.

## See Also

- [check](check.md) - Environment diagnosis
- [install](install.md) - Get safe install commands
