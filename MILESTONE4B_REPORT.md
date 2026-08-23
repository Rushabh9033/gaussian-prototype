# MILESTONE 4B: BOUNDED ALTERNATIVE ARCHITECTURE BAKE-OFF

**GOAL:** Find whether a fundamentally different architecture can generate five coherent, sharp, synthetic deep-zoom levels from one normal photograph on an 8.59GB VRAM GPU.

## HARDWARE ENVIRONMENT
* GPU: NVIDIA GeForce RTX 4060 Laptop GPU
* VRAM: 8.59 GB
* Framework: PyTorch 2.6.0+cu124 (Python 3.13)

## EXPERIMENT STATUS: ALL CANDIDATES BLOCKED

### CANDIDATE A: Chain-of-Zoom (arXiv:2412.10300)
**Status:** `BLOCKED`
**Reason:** Severe Hardware VRAM limitations. The official implementation dictates approximately 24GB VRAM is required to load `stabilityai/stable-diffusion-3-medium-diffusers` + `Qwen/Qwen2.5-VL-7B-Instruct` + `xinyu1205/recognize-anything-plus-model` concurrently, even when utilizing their efficient-memory mode. Our hardware only possesses 8.59GB.

### CANDIDATE B: Generative Powers of Ten (arXiv:2312.02149)
**Status:** `BLOCKED_NO_REPRODUCIBLE_IMPLEMENTATION`
**Reason:** The authors did not release official code. The primary unofficial replication requires DeepFloyd IF, which mandates a gated HuggingFace license and is known to consume massive amounts of VRAM that exceed our 8.59GB budget.

### CANDIDATE C: Direct-from-Root Structure-Conditioned (SD1.5 ControlNet)
**Status:** `BLOCKED`
**Reason:** Checkpoint / Environment Failure.
**Command Executed:** `python run_m4b_bakeoff.py`
**Error Evidence:**
1. Checkpoint network streams (`Salesforce/blip-image-captioning-base` and `lllyasviel/control_v11f1e_sd15_tile`) hang indefinitely during transfer, failing to write the large `.bin` files to disk.
2. When forced or falling back, the Python 3.13 runtime suffers a hard C++ segmentation fault (Access Violation) when parsing legacy pickle weights (`.bin`).
Event Viewer Trace:
```text
Faulting application name: python.exe, version: 3.13.7150.1013
Exception code: 0xc0000005
Fault offset: 0x0000000000000001
Faulting process id: 0x632c
```
Due to the unavailability of `safetensors` for these specific legacy models on the Hub and the resulting environment hard-crash, the pipeline is blocked from initializing.
