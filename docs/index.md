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

**I just wasted 3 hours fighting CUDA errors that broke my GPU environment! Been there?**

---

> **"Why does my PyTorch crash with CUDA errors when I just installed it?"**
>
> Because your driver supports CUDA 11.8, but `pip install torch` gave you CUDA 12.4 wheels.

---

Env-Doctor diagnoses and fixes the #1 frustration in GPU computing: mismatched CUDA versions between your NVIDIA driver, system toolkit (nvcc compiler), cuDNN, and Python libraries.

It takes **5 seconds** to find out if your environment is broken — and exactly how to fix it.

## Features

| Feature | What It Does |
|---------|--------------|
| **One-Command Diagnosis** | Instantly check compatibility between GPU Driver → CUDA Toolkit → cuDNN → PyTorch/TensorFlow/JAX |
| **CUDA Installation Guide** | Get platform-specific, copy-paste CUDA installation commands for Ubuntu, Debian, RHEL, Fedora, WSL2, Windows, and Conda |
| **Deep CUDA Analysis** | Reveals multiple installations, PATH issues, environment misconfigurations |
| **Compilation Guard** | Warns if system `nvcc` doesn't match PyTorch's CUDA — preventing flash-attention build failures |
| **WSL2 GPU Support** | Detects WSL1/WSL2 environments, validates GPU forwarding, catches common driver conflicts for WSL2 |
| **Safe Install Commands** | Prescribes the exact `pip install` command that works with YOUR driver |
| **Container Validation** | Catches GPU config errors in Dockerfiles and docker-compose with DB-driven recommendations |
| **AI Model Compatibility** | Check if your GPU can run any model (LLMs, Diffusion, Audio) before downloading |
| **cuDNN Detection** | Finds cuDNN libraries, validates symlinks, checks version compatibility |
| **Migration Helper** | Scans code for deprecated imports (LangChain, Pydantic) and suggests fixes |



## Installation

```bash
pip install env-doctor
```

## Quick start (Other commands explained extensively seperately)

```bash
# Diagnose your environment
env-doctor check

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
