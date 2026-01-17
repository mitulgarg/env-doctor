# Env-Doctor

**The missing link between your GPU and Python AI libraries**

---

> **"Why does my PyTorch crash with CUDA errors when I just installed it?"**
>
> Because your driver supports CUDA 11.8, but `pip install torch` gave you CUDA 12.4 wheels.

---

Env-Doctor diagnoses and fixes the #1 frustration in GPU computing: mismatched CUDA versions between your NVIDIA driver, system toolkit, cuDNN, and Python libraries.

It takes **5 seconds** to find out if your environment is broken — and exactly how to fix it.

## Quick Demo

<!-- Replace YOUR_RECORDING_ID with your Asciinema recording ID -->
<!--
<script src="https://asciinema.org/a/YOUR_RECORDING_ID.js" id="asciicast-YOUR_RECORDING_ID" async data-autoplay="true" data-loop="true" data-speed="1.5"></script>
-->

*Demo coming soon - see [Recording Demos](guides/recording-demos.md) for instructions*

## Model Checker Demo

<!-- Replace YOUR_MODEL_DEMO_ID with your Asciinema recording ID -->
<!--
<script src="https://asciinema.org/a/YOUR_MODEL_DEMO_ID.js" id="asciicast-YOUR_MODEL_DEMO_ID" async data-autoplay="true" data-speed="1.5"></script>
-->

*Model compatibility demo coming soon*

## Features

| Feature | What It Does |
|---------|--------------|
| **One-Command Diagnosis** | Instantly check compatibility between GPU Driver → CUDA Toolkit → cuDNN → PyTorch/TensorFlow/JAX |
| **Deep CUDA Analysis** | Reveals multiple installations, PATH issues, environment misconfigurations |
| **cuDNN Detection** | Finds cuDNN libraries, validates symlinks, checks version compatibility |
| **Container Validation** | Catches GPU config errors in Dockerfiles and docker-compose with DB-driven recommendations |
| **AI Model Compatibility** | Check if your GPU can run any model (LLMs, Diffusion, Audio) before downloading |
| **WSL2 GPU Support** | Detects WSL1/WSL2 environments, validates GPU forwarding, catches common driver conflicts |
| **Compilation Guard** | Warns if system `nvcc` doesn't match PyTorch's CUDA — preventing flash-attention build failures |
| **Safe Install Commands** | Prescribes the exact `pip install` command that works with YOUR driver |
| **Migration Helper** | Scans code for deprecated imports (LangChain, Pydantic) and suggests fixes |

## Installation

```bash
pip install env-doctor
```

Or install from source:

```bash
git clone https://github.com/mitulgarg/env-doctor.git
cd env-doctor
pip install -e .
```

## Quick Start

```bash
# Diagnose your environment
env-doctor check

# Get safe install command for PyTorch
env-doctor install torch

# Check if a model fits on your GPU
env-doctor model llama-3-8b
```

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
