# Milestone 6A: Classical Multi-Frame Super-Resolution — Controlled Feasibility

## 1. Architecture

```
multiframe_sr/
  forward_model.py     — sRGB↔linear, blur, downsample, warp
  frame_generator.py   — deterministic sub-pixel shifted frame synthesis
  registration.py      — phase correlation + ECC refinement
  robust_fusion.py     — shift-and-add baseline + Huber-weighted iterative fusion
  backprojection.py    — iterative back-projection with Laplacian regularization
  metrics.py           — PSNR, SSIM, edge F1, gradient energy, ringing, etc.
```

## 2. Mathematical Mechanism

### Why Multiple Frames Contain Additional Information

A single low-resolution pixel is a spatial average (integral) of the high-resolution scene weighted by the point spread function (PSF) and the pixel sampling grid. When multiple images are captured with different sub-pixel offsets, each observation samples different high-frequency content from the scene. The Nyquist-Shannon theorem implies that if the LR sampling rate is below the scene's bandwidth, information is aliased and lost in any single frame — but different sub-pixel positions alias *different* frequency components. By combining N frames with distinct offsets, we recover up to N× more spatial frequency information than any single frame contains.

The reconstruction proceeds as:
1. **Registration**: Phase correlation provides initial shift estimates; Enhanced Correlation Coefficient (ECC) refines to sub-pixel accuracy.
2. **Robust Fusion**: Each frame is placed on a high-resolution grid at its estimated position. A Huber-weighted iterative mean suppresses outliers and occlusion artifacts.
3. **Back-Projection**: The forward model (warp → blur → downsample) is applied to the current HR estimate, the residual against each observed LR frame is computed, and the inverse-projected residual gradient updates the HR estimate under Laplacian regularization.

## 3. Exact Commands

```
python run_m6a_benchmark.py
python validate_m6a.py
python -m pytest tests -v
npm run build  (in src/viewer)
```

## 4. Registration Results

| Scale | Frames | Accepted | Rejected | Registration RMSE (LR px) |
|-------|--------|----------|----------|--------------------------|
| 2×    | 8      | 8        | 0        | 0.0191                   |
| 4×    | 16     | 16       | 0        | 0.0205                   |

All frames accepted. Sub-pixel alignment was estimated without access to ground-truth transformations.

## 5. Measurements

### 2× Scale (8 frames → 360×220)

- Lanczos PSNR: 24.2398 dB
- Multi-Frame SR PSNR: 24.2481 dB
- PSNR gain: +0.0084 dB
- Lanczos SSIM: 0.7560
- Multi-Frame SR SSIM: 0.7571
- SSIM gain: +0.0010
- Edge F1 change: −0.0108

### 4× Scale (16 frames → 360×220)

- Lanczos PSNR: 21.0308 dB
- Multi-Frame SR PSNR: 21.0453 dB
- PSNR gain: +0.0145 dB
- Lanczos SSIM: 0.5401
- Multi-Frame SR SSIM: 0.5427
- SSIM gain: +0.0026
- Edge F1 change: −0.0332

### Memory

- Maximum measured Python allocation: approximately 122.46 MB

## 6. Limitations

1. **This is a controlled experiment.** The LR frames were synthetically generated from a known HR ground truth with known sub-pixel translations. Real-world multi-frame SR faces motion blur, varying illumination, occlusion, and non-translational motion.
2. **Translation-only model.** The registration and reconstruction assume pure translation. Real cameras produce rotation, zoom, and lens distortion.
3. **The original single-image vision has NOT been achieved.** This milestone evaluates whether multiple independently sampled observations of the same scene can recover additional spatial information that is absent from any single observation.
4. **EVIDENCE LIMITATION:** The benchmark used committed code `3af56b61b7a17ffdd43c9fb7974831a4f7d8c61b`, but the run did not begin with a clean tree because previously tracked evidence files had been removed before execution.

## 7. Test Results

- **pytest**: 112 passed, 0 failed (includes all M4D, M5, M5B, M5C, M5D, M6A tests)
- **Viewer build**: exit code 0

## 8. Final Status

**NO_MEASURABLE_GAIN**

Registration succeeded and multiple frames were aligned correctly. However, Robust Fusion produced only a very small quality change. The corrected iterative back-projection algorithm accepted no useful improvement over the baseline constraints and returned results identical to Robust Fusion.

As a result, the required PSNR, SSIM, and edge thresholds were not achieved. The experiment does not achieve the original single-image vision. Milestone 6A is closed as a completed negative experiment.
