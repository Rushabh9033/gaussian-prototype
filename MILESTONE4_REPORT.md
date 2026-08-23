# Milestone 4 — Deterministic Baked Generated Branch

## Status: RESEARCH FAILED

**MILESTONE 4 REPRESENTATION WORKS, BUT RECURSIVE X4 GENERATION QUALITY FAILED.**

The V3 representation (format, bounding boxes, metadata, progressive loading, strict validation) works flawlessly and is completely implemented and tested.

However, recursive x4 AI generation using `stabilityai/stable-diffusion-x4-upscaler` failed the required deep-zoom quality. It cannot retain deep-zoom photographic sharpness and microstructural detail across 5 levels (1024x cumulative zoom) without external conditioning or a different architectural approach.

### Bounded Experiment Results

We ran three bounded configurations applying progressively higher generative noise (`noise_level`) across 5 levels (20 steps each). Configs A, B, and C were the final bounded experiments and no further tuning is authorized.

- **Config A** (Noise 20, 20, 20, 20, 20): Gradient Energy dropped severely from Level 1 to Level 5. Result: Smooth blurred surface.
- **Config B** (Noise 20, 30, 40, 50, 60): Gradient Energy still dropped heavily. Result: Blur with slight noise.
- **Config C** (Noise 20, 40, 60, 80, 100): Detail loss is still >90%, failing to preserve meaningful microstructure or photographic sharpness.

High parent SSIM values in earlier levels confirmed that the child successfully matched the parent, but because the x4 upscaler inherently smooths out unresolved high-frequency detail at these extreme synthetic magnifications, the parent itself becomes progressively blurrier, leading to a cascade of detail loss.

The approach is marked as a **FAILED RESEARCH APPROACH**. No further generation or prompt/noise tuning will be performed.
