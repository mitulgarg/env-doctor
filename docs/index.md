# Env-Doctor

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "Env-Doctor",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Linux, Windows, macOS",
  "description": "The missing link between your GPU and Python AI libraries. Diagnose and fix CUDA version mismatches, validate Docker GPU configurations, and check AI model compatibility.",
  "url": "https://mitulgarg.github.io/env-doctor/",
  "downloadUrl": "https://pypi.org/project/env-doctor/",
  "softwareVersion": "latest",
  "author": {
    "@type": "Person",
    "name": "Mitul Garg"
  },
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "5",
    "ratingCount": "1"
  },
  "keywords": "CUDA version mismatch, PyTorch CUDA error, GPU diagnostics, NVIDIA driver compatibility, torch not compiled with CUDA, cuDNN error, flash-attention build failed, docker GPU troubleshooting, WSL2 GPU, AI model VRAM calculator, LLM GPU requirements, GPU environment setup, CUDA debugging tool",
  "codeRepository": "https://github.com/mitulgarg/env-doctor",
  "programmingLanguage": "Python",
  "softwareRequirements": "Python 3.7+",
  "license": "https://opensource.org/licenses/MIT",
  "potentialAction": {
    "@type": "InstallAction",
    "target": {
      "@type": "EntryPoint",
      "urlTemplate": "pip install env-doctor",
      "actionPlatform": [
        "http://schema.org/DesktopWebPlatform",
        "http://schema.org/IOSPlatform",
        "http://schema.org/AndroidPlatform"
      ]
    }
  }
}
</script>

<div class="badge-row">
  <a href="https://pypi.org/project/env-doctor/"><img src="https://img.shields.io/pypi/v/env-doctor?style=flat-square&labelColor=161b22&color=006d32&label=PyPI" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/env-doctor/"><img src="https://img.shields.io/pypi/dm/env-doctor?style=flat-square&labelColor=161b22&color=006d32&label=Downloads%2Fmonth" alt="Monthly Downloads"></a>
  <a href="https://github.com/mitulgarg/env-doctor/stargazers"><img src="https://img.shields.io/github/stars/mitulgarg/env-doctor?style=flat-square&labelColor=161b22&color=006d32" alt="GitHub Stars"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-006d32?style=flat-square&labelColor=161b22" alt="Python 3.10+">
  <a href="https://github.com/mitulgarg/env-doctor/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mitulgarg/env-doctor?style=flat-square&labelColor=161b22&color=006d32" alt="License"></a>
</div>

**Env-Doctor is a GPU environment diagnostic tool for Python AI/ML developers.**

It detects and fixes the most common source of broken GPU setups: version mismatches between your NVIDIA driver, CUDA toolkit, cuDNN, and Python libraries like PyTorch, TensorFlow, and JAX.

Run one command. Get a full diagnosis. Get the exact fix.

```bash
pip install env-doctor
env-doctor check
```

> **Common symptom:** `torch.cuda.is_available()` returns `False` right after installing PyTorch — because your driver supports CUDA 11.8, but `pip install torch` silently pulled CUDA 12.4 wheels.

Env-Doctor also checks GPU architecture compatibility, Python version conflicts, Docker GPU configs, AI model VRAM requirements, and exposes all diagnostics to AI assistants via a built-in MCP server.

## Features

| Feature | What It Does |
|---------|--------------|
| **One-Command Diagnosis** | Instantly check compatibility between GPU Driver → CUDA Toolkit → cuDNN → PyTorch/TensorFlow/JAX |
| **Compute Capability Check** | Detect GPU architecture mismatches — catches why `torch.cuda.is_available()` returns `False` on new GPUs (e.g. Blackwell RTX 5000) even when driver and CUDA are healthy |
| **Python Version Compatibility** | Detect Python version conflicts with AI libraries and dependency cascade impacts |
| **CUDA Installation Guide** | Get platform-specific, copy-paste CUDA installation commands for Ubuntu, Debian, RHEL, Fedora, WSL2, Windows, and Conda |
| **Deep CUDA Analysis** | Reveals multiple installations, PATH issues, environment misconfigurations |
| **Compilation Guard** | Warns if system `nvcc` doesn't match PyTorch's CUDA — preventing flash-attention build failures |
| **WSL2 GPU Support** | Detects WSL1/WSL2 environments, validates GPU forwarding, catches common driver conflicts for WSL2 |
| **Safe Install Commands** | Prescribes the exact `pip install` command that works with YOUR driver |
| **Container Validation** | Catches GPU config errors in Dockerfiles and docker-compose with DB-driven recommendations |
| **AI Model Compatibility** | Check if your GPU can run any model (LLMs, Diffusion, Audio) before downloading |
| **cuDNN Detection** | Finds cuDNN libraries, validates symlinks, checks version compatibility |
| **MCP Server** | Expose all diagnostics to AI assistants (Claude Code, Claude Desktop, Zed) via Model Context Protocol stdio — no browser needed |
| **Migration Helper** | Scans code for deprecated imports (LangChain, Pydantic) and suggests fixes |



## Installation

```bash
pip install env-doctor
```

## Quick start (Other commands explained extensively seperately)

```bash
# Diagnose your environment
env-doctor check

# Check Python version compatibility
env-doctor python-compat

# Get CUDA installation instructions
env-doctor cuda-install

# Get safe install command for PyTorch
env-doctor install torch

# Check if a model fits on your GPU
env-doctor model llama-3-8b
```


## Quick Sneak Peek (There's WAY more to Env-Doctor)

<!-- For future updates, Replace YOUR_RECORDING_ID with your Asciinema recording ID -->

<div style="max-width: 700px; margin: auto;">

<script src="https://asciinema.org/a/0OBygpGsyreSfn1c.js" id="asciicast-0OBygpGsyreSfn1c" async data-autoplay="true" data-loop="true" data-speed="1.5"></script>
</div>


## Model Checker Demo

<!-- For future updates, Replace YOUR_MODEL_DEMO_ID with your Asciinema recording ID -->

<div style="max-width: 700px; margin: auto;">

<script src="https://asciinema.org/a/ilS2kEWDY5SeVVE4.js" id="asciicast-ilS2kEWDY5SeVVE4" async data-autoplay="true" data-speed="1.5"></script>

</div>



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mitulgarg/env-doctor&type=Date)](https://star-history.com/#mitulgarg/env-doctor&Date)

## What's Next?

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Getting Started**

    ---

    Complete installation and first steps guide

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-console:{ .lg .middle } **Commands**

    ---

    Full reference for all CLI commands

    [:octicons-arrow-right-24: Commands](commands/check.md)

-   :fontawesome-brands-docker:{ .lg .middle } **Container Validation**

    ---

    Validate Dockerfiles and docker-compose for GPU issues

    [:octicons-arrow-right-24: Dockerfile Validation](commands/dockerfile.md)

-   :material-robot:{ .lg .middle } **Model Compatibility**

    ---

    Check if AI models fit on your GPU before downloading

    [:octicons-arrow-right-24: Model Checker](commands/model.md)

</div>

---

## Frequently Asked Questions

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "Why does torch.cuda.is_available() return False?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "The most common causes are: (1) your GPU's compute capability (SM architecture) is not included in your installed PyTorch wheel — common on new GPUs like Blackwell RTX 5000 series; (2) your NVIDIA driver version is too old for your CUDA toolkit; (3) there is a CUDA version mismatch between your driver, toolkit, and PyTorch. Run 'env-doctor check' to diagnose the exact cause and get a fix command."
      }
    },
    {
      "@type": "Question",
      "name": "How do I fix a CUDA version mismatch with PyTorch?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Run 'env-doctor check' to identify which versions are mismatched. Then run 'env-doctor install torch' to get the exact pip install command with the correct --index-url for your driver. Env-doctor automatically matches the CUDA version supported by your installed NVIDIA driver."
      }
    },
    {
      "@type": "Question",
      "name": "How do I check if my NVIDIA driver is compatible with my CUDA toolkit?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Run 'env-doctor check'. It reads your driver version, calculates the maximum CUDA version it supports, and compares that against your installed CUDA toolkit and PyTorch CUDA build. Any incompatibilities are shown with specific fix instructions."
      }
    },
    {
      "@type": "Question",
      "name": "Why does flash-attention fail to build?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "flash-attn requires an exact match between your system nvcc (CUDA toolkit) version and the CUDA version PyTorch was compiled with. Run 'env-doctor install flash-attn' to detect the mismatch and get two options: downgrade PyTorch to match your nvcc, or upgrade your CUDA toolkit to match PyTorch."
      }
    },
    {
      "@type": "Question",
      "name": "How do I use env-doctor with Claude or other AI assistants?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Env-doctor includes a built-in MCP (Model Context Protocol) server. After installing env-doctor, add it to your Claude Desktop or Claude Code config as 'env-doctor-mcp'. Your AI assistant can then call env-doctor's diagnostic tools directly — checking your GPU environment, fetching safe install commands, and validating Dockerfiles — without you leaving the chat."
      }
    },
    {
      "@type": "Question",
      "name": "Does env-doctor work on Windows and WSL2?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes. Env-doctor detects WSL1 vs WSL2 automatically and validates GPU forwarding in WSL2. It also supports native Windows with CUDA detection and provides Windows-specific CUDA installation instructions via 'env-doctor cuda-install'."
      }
    },
    {
      "@type": "Question",
      "name": "My new RTX 5000 / Blackwell GPU is not working with PyTorch. What should I do?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Stable PyTorch wheels do not yet include SM 120 (Blackwell) support. Run 'env-doctor check' — it will detect the architecture mismatch and tell you whether it is a hard failure (is_available() returns False) or a soft failure (PTX JIT workaround active but ops may degrade). It will then provide the exact nightly PyTorch install command with sm_120 support."
      }
    }
  ]
}
</script>

??? question "Why does `torch.cuda.is_available()` return `False`?"
    The most common causes are:

    1. Your GPU's SM architecture isn't in your PyTorch wheel (common on new GPUs like Blackwell RTX 5000)
    2. Your NVIDIA driver is too old for your CUDA toolkit
    3. CUDA version mismatch between driver, toolkit, and PyTorch

    Run `env-doctor check` to get the exact cause and fix.

??? question "How do I fix a CUDA version mismatch with PyTorch?"
    Run `env-doctor check` to identify what's mismatched, then `env-doctor install torch` to get the exact `pip install` command with the correct `--index-url` for your driver.

??? question "Why does flash-attention fail to build?"
    flash-attn requires an exact match between your system `nvcc` version and PyTorch's CUDA build. Run `env-doctor install flash-attn` — it detects the mismatch and gives you two fix paths.

??? question "How do I use env-doctor with Claude or other AI assistants?"
    Env-doctor ships a built-in MCP server (`env-doctor-mcp`). Add it to your Claude Desktop or Claude Code config and your AI assistant can call all diagnostic tools directly from the chat. See the [MCP Integration Guide](guides/mcp-integration.md).

??? question "My new RTX 5000 / Blackwell GPU isn't working with PyTorch. What do I do?"
    Stable PyTorch wheels don't yet include SM 120 (Blackwell) support. Run `env-doctor check` — it detects whether you have a hard or soft architecture mismatch and provides the exact nightly PyTorch install command with `sm_120` support.
