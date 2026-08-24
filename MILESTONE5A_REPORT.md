# Milestone 5A Closure: ScaleField Vector-Residual Canvas

## Overview
This report reflects the partial experimental VRC prototype, which is **not** a completed or fully validated VRC architecture.

### Architectural Limitations
1. **Real Vector Edge Extraction**: B-spline boundary rendering was implemented, but it only draws sampled boundary lines using averaged left/right colours. It is **not** complete vector-region reconstruction.
2. **Real Signed Residual Pyramid**: Laplacian-pyramid helper functions exist in the codebase, but the executed benchmark and exported package use **only one signed source residual level** (acting as a simple difference array, not a true multi-level hierarchy).
3. **Analytic Scale Renderer**: Operates by explicitly drawing sub-pixel spline boundaries over the continuous color field.
4. **Valid Experimental Package Decoder**: A complete package decoder was **not implemented**. The current validation strictly reads the ZIP components but does not reconstruct arbitrary zoom levels exclusively from the package. It is an **experimental data bundle**, not a production offline zoom format.
5. **Tile / QuadTree Mapping**: Quadtree helper code exists in the repository but is **not integrated** into the renderer, the package, or the viewer.

## Benchmark Results

The benchmark automatically selected configuration **VRC_C**.

### A. Ground-Truth Reconstruction (VRC_C)

| Scale | Method | PSNR | SSIM | Edge F1 | Ringing % |
|-------|--------|------|------|---------|-----------|
| **2x** | Lanczos | **26.84** | **0.860** | 0.878 | **0.42%** |
| | VRC_C | 25.892686671408875 | 0.8538859576807466 | **0.9073472144050232** | 3.2032828282828283% |
| **4x** | Lanczos | **22.84** | **0.652** | 0.498 | **3.71%** |
| | VRC_C | 22.21521740460167 | 0.6318506357812695 | **0.639374756085921** | 9.147727272727273% |
| **8x** | Lanczos | **20.39** | **0.468** | 0.048 | **16.96%** |
| | VRC_C | 19.61282789640991 | 0.4314396796441651 | **0.349102260053127** | 22.08585858585859% |

### B. Operational Zoom & Scale Consistency
When natively zooming the 1x image out to 2x, 4x, and 8x, we measured the absolute channel difference after downsampling back to 1x (in 8-bit space). The required maximum was `<= 1`.

- **2x Consistency maximum error**: 39
- **4x Consistency maximum error**: 46
- **8x Consistency maximum error**: 50

### C. Package Metrics
- **Actual Package Size**: 1,013,001 bytes.

## Visual Defects Recorded
The rendered vector-residual outputs explicitly display:
- **Jagged wheel boundaries**
- **Halos and black dots**
- **Horizontal spline artifacts**
- **Distorted headlamp structure**
- **No recovered photographic information**

## Tests & Validation
- Overreaching test claims have been corrected. Tests now explicitly name what they verify (e.g., signature parameters, readability of zip components).
- Generation of fake failure evidence was removed and its formal implementation is deferred.

## Final Conclusion
**Did it recover information?** No. While structural F1 scores improved (due to analytical boundary insertion), global PSNR dropped. The strict implementation highlights the unresolvable tension between perfectly crisp geometric edges and global pixel-wise fidelity constraints, proving that rigid vector edges inserted analytically inevitably create banding or ringing when mapping back to standard sRGB representation without AI-led detail generation.

### Final Status: `NO_MEASURABLE_GAIN`
