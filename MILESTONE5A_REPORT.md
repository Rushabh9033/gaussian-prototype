# Milestone 5A: ScaleField Vector-Residual Canvas

## Overview
This milestone tests a pure-mathematical proof of concept for the ScaleField Vector-Residual Canvas (VRC). The method proposes representing an image as a continuous-coordinate color field supplemented by vector-like structural boundaries (B-splines) and source-supported mathematical residuals. 

All algorithms operate deterministically without AI, neural networks, or pretrained models.

### Reused Architecture
- Existing Python project conventions
- Color space conversion utilities (sRGB to linear and back)
- V3 package hashing and metadata utilities
- Milestone 4D metrics and benchmarking framework
- Existing `porche.png` source and structural coordinates
- Existing viewer production build

### New Experimental Modules (`src/python/m5_vrc/`)
1. **`vector_edges.py`**: Calculates the structure tensor in linear luminance, computes edge coherence, traces strong edges, and fits parametric B-splines.
2. **`color_field.py`**: Evaluates a deterministic continuous color field using a bilateral filter to prevent blending across high-confidence boundaries.
3. **`residual_pyramid.py`**: Computes and evaluates the mathematical residual ($R = Source - Base$). No high-frequency bands are invented.
4. **`scale_renderer.py`**: Synthesizes the components at arbitrary scales and enforces the iterative parent-child scale consistency constraint via residual distribution.
5. **`quadtree.py`**: Divides output grids into deterministic spatial tiles for localized rendering.

## Benchmark Results
The benchmark executed against three conservative structural configurations (A, B, C). Configuration **VRC_B** (Coherence > 0.5, Strength > 0.10) was automatically selected based on optimal F1 structural scores.

### A. Ground-Truth Reconstruction

| Scale | Method | PSNR | SSIM | Edge F1 | Ringing % |
|-------|--------|------|------|---------|-----------|
| **2x** | Lanczos | **26.84** | **0.860** | 0.878 | **0.42%** |
| | VRC_B | 25.86 | 0.857 | **0.907** | 3.06% |
| **4x** | Lanczos | **22.84** | **0.652** | 0.498 | **3.71%** |
| | VRC_B | 22.34 | 0.649 | **0.616** | 7.41% |
| **8x** | Lanczos | **20.39** | **0.468** | 0.048 | **16.96%** |
| | VRC_B | 19.90 | 0.455 | **0.256** | 19.53% |

### B. Operational Zoom (Scale Consistency)
During native zooming (360x220 -> 2880x1760), the iterative consistency algorithm failed to fully suppress aliasing errors at sharp structure boundaries within the permitted 5 iterations:
- **2x Max Channel Error**: 38.0
- **4x Max Channel Error**: 38.0
- **8x Max Channel Error**: 36.0

## Artifacts and Deliverables
- **VRC Package**: `m5_results/vrc_package.zip` (104 KB)
- **Results JSON**: `m5_results/results.json`
- **Execution Evidence**: `m5_results/evidence.jsonl`
- **Contact Sheets**: `m5_results/contact_2x_wheel.png`, `m5_results/contact_8x_headlamp.png`, etc.
- **Fitted Curve Overlay**: `m5_results/fitted_curve_overlay.png`

## Artifact Hashes
Available in `m5_results/evidence.jsonl`. Source Hash: `8eda0f62759c5d72da363eb338329a87650cd931076fb1c0f932d71dbefdafc1`.

## Tests & Viewer Build
- **Tests**: All 47 CPU-only Python tests passed. Verified scale invariance, deterministic spline fitting, accurate residual reconstruction, artifact hashing, absent AI dependencies, and pure origin-to-scale rendering without intermediate bitmap re-upscaling.
- **Viewer**: The production viewer compiled successfully in 96ms (0 vulnerabilities).

## Limitations and Conclusion
**Did it recover information?** No. VRC exclusively reconstructed structure using boundaries parameterized from the source itself. 

While VRC significantly outperformed Lanczos on structural edge retention (Edge F1), the discrete analytic rasterization of B-splines over the color field introduced noise that severely penalized global PSNR (loss of ~1.0 dB). Furthermore, the scale-consistency constraint could not reliably stay within the stringent maximum bounds near mathematical edges.

### Final Status: `NO_MEASURABLE_GAIN`
The experiment executed cleanly, but the vector-residual canvas does not provide a globally defensible mathematical improvement over Lanczos. The limitations of non-AI classical interpolation hold.
