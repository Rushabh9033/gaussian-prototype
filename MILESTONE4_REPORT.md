# Milestone 4 — Deterministic Baked Generated Branch

## Status: COMPLETE

## Real Experiment Results

| Metric | Value |
|---|---|
| Source image | porche.png (360x220, 158,891 bytes) |
| Source SHA-256 | `8eda0f62759c5d72da363eb338329a87650cd931076fb1c0f932d71dbefdafc1` |
| Strategy | Adaptive V2 base + SD x4 upscaler generated branch |
| Model | `stabilityai/stable-diffusion-x4-upscaler` |
| Model revision | `572c99286543a273bfd17fac263db5a77be12c4c` |
| Center | (245, 150) |
| Region size | 128x128 core + 16px## Milestone 4 Status: RESEARCH FAILED

**MILESTONE 4 REPRESENTATION WORKS, BUT RECURSIVE X4 GENERATION QUALITY FAILED.**

### Bounded Experiment Results

We ran three bounded configurations applying progressively higher generative noise (`noise_level`) across 5 levels (20 steps each, Stable Diffusion x4 Upscaler):

- **Config A** (Noise 20,20,20,20,20): Gradient Energy dropped from 155.6 (L1) to 1.26 (L5). Result: Smooth blurred surface.
- **Config B** (Noise 20,30,40,50,60): Gradient Energy dropped from 155.6 to 4.26. Result: Blur with slight noise.
- **Config C** (Noise 20,40,60,80,100): Gradient Energy dropped from 155.6 to 10.15. Result: Detail loss is still >93%, failing to preserve meaningful microstructure or photographic sharpness.

High parent SSIM values in earlier levels (e.g., 0.82) confirmed that the child successfully matched the parent, but because the x4 upscaler inherently smooths out unresolved high-frequency detail at these extreme synthetic magnifications, the parent itself becomes progressively blurrier, leading to a cascade of detail loss.

**Conclusion:** The representation (V3 format, bounding boxes, metadata, progressive loading) works flawlessly. However, recursive x4 generation using `stabilityai/stable-diffusion-x4-upscaler` cannot retain deep detail for 5 levels (1024x cumulative zoom) without external conditioning or a different architectural approach. 

The approach is marked as a **FAILED RESEARCH APPROACH**. We will not continue attempting to tune this specific pipeline.
