# Milestone 5A Correction: ScaleField Vector-Residual Canvas

## Overview
This report reflects the completed, fully validated non-AI ScaleField Vector-Residual Canvas architecture. The method has been corrected to genuinely implement mathematical curve rendering and signed residual pyramids, adhering strictly to the classical limits of the original image without any AI generation.

### Architecture Implemented
1. **Real Vector Edge Extraction**: Sub-pixel precision gradients mapping to exact parameterized B-splines. Deterministic structure tensor localization, non-maximum suppression (NMS), curve fitting, left/right normal boundary color sampling, and fit-error rejection.
2. **Real Signed Residual Pyramid**: A strict Laplacian pyramid storing the true $R = Source - Base$ difference in signed, high-precision numeric formats (numpy `.npz`), avoiding lossy PNG clipping.
3. **Analytic Scale Renderer**: Scaled renderings now evaluate the exact parametric B-spline math coordinates onto the higher-resolution lattice, drawing explicit sub-pixel boundary geometry rather than relying solely on rasterized pixel interpolation.
4. **Valid Experimental Package Decoder**: The package now round-trips without `pickle`. It parses pure JSON parameters and `np.savez` residuals, decoding and validating internal component hashes correctly.
5. **Tile / QuadTree Mapping**: True spatial quadtree bounds indexing and viewport-based visible tile selection.

## Benchmark Results

The benchmark generated ground-truth restorations at 2x, 4x, and 8x scale using three non-AI configurations. **VRC_B** was selected dynamically based on optimal Edge F1 constraints vs acceptable PSNR loss against Lanczos.

### A. Ground-Truth Reconstruction

| Scale | Method | PSNR | SSIM | Edge F1 | Ringing % |
|-------|--------|------|------|---------|-----------|
| **2x** | Lanczos | **26.84** | **0.860** | 0.878 | **0.42%** |
| | VRC_B | 25.72 | 0.848 | **0.905** | 3.55% |
| **4x** | Lanczos | **22.84** | **0.652** | 0.498 | **3.71%** |
| | VRC_B | 22.02 | 0.617 | **0.646** | 10.55% |
| **8x** | Lanczos | **20.39** | **0.468** | 0.048 | **16.96%** |
| | VRC_B | 19.72 | 0.435 | **0.342** | 21.29% |

*VRC_B achieved up to 14.5% vector coverage and ~0.16 curve fitting error at 2x.*

### B. Operational Zoom & Scale Consistency
When natively zooming the 1x image out to 2x, 4x, and 8x, we measured the absolute channel difference after downsampling back to 1x (in 8-bit space).
- **2x Consistency**: Mean Error: 0.58 | Max Error: 39.0
- **4x Consistency**: Mean Error: 0.83 | Max Error: 46.0
- **8x Consistency**: Mean Error: 0.88 | Max Error: 50.0

Due to rigid pixel clipping and boundary out-of-gamut overshoots introduced by analytic splines, the maximum absolute consistency error remained stubbornly high (~50 in 8-bit space).

## Artifacts and Deliverables
- **VRC Package**: `m5_results/vrc_package.zip` (~328 KB). Successfully decodes logic, spline geometries, and signed residual differences safely without `pickle`.
- **Results JSON**: `m5_results/results.json`
- **Execution Evidence**: `m5_results/evidence.jsonl`
- Ground-truth reconstructed benchmarks and contact sheets correctly mapped for 2x, 4x, and 8x.

## Tests & Validation
- **Tests**: 44 CPU-only Python tests strictly verifying the true parameters of the implementation. Bare `pass` blocks were replaced with explicit signature, execution, and deterministic hierarchy tests.
- **Artifact Hashing**: The suite generates and verifies hard SHA-256 validation sums.
- **Viewer**: The production viewer compiled successfully in <100ms.

## Final Conclusion
**Did it recover information?** No. While structural F1 scores heavily improved (thanks to analytical B-spline sharp overlays), the rigid geometric insertions visibly deteriorated global PSNR (loss of ~1dB) and resulted in halo/ringing artifacts, confirming that it simply swaps blurry structure (Lanczos) for jagged/ringing structure (VRC) without actually inventing new photographic detail. 

### Final Status: `NO_MEASURABLE_GAIN`
The execution correctly implemented the mathematical architecture, yet it decisively failed to mathematically overcome Lanczos overall. The constraints of non-AI scaling remain confirmed.
