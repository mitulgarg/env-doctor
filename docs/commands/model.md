# model

Check if an AI model fits on your GPU before downloading.

## Usage

```bash
env-doctor model <model-name>
```

## Options

| Option | Description |
|--------|-------------|
| `--list` | List all available models |
| `--precision <type>` | Check specific precision (fp32, fp16, int8, int4) |

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

## Supported Model Categories

### LLMs

Large Language Models for text generation:

| Model | Parameters | FP16 VRAM |
|-------|------------|-----------|
| Llama-3-8B | 8B | ~19GB |
| Llama-3-70B | 70B | ~140GB |
| Mistral-7B | 7B | ~16GB |
| Mixtral-8x7B | 46.7B | ~93GB |

### Diffusion Models

Image generation models:

| Model | VRAM (FP16) |
|-------|-------------|
| Stable Diffusion 1.5 | ~4GB |
| Stable Diffusion XL | ~8GB |
| Flux Schnell | ~12GB |

### Audio Models

Speech recognition and synthesis:

| Model | VRAM |
|-------|------|
| Whisper Tiny | ~1GB |
| Whisper Large | ~10GB |

## Model Aliases

Common aliases are supported:

- `sdxl` → `stable-diffusion-xl`
- `sd15` → `stable-diffusion-1.5`
- `llama3` → `llama-3-8b`

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
