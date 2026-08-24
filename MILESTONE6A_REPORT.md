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

| Method | PSNR (dB) | SSIM | Edge F1 | Ringing % |
|--------|-----------|------|---------|-----------|
| Lanczos (single frame) | 24.24 | 0.7560 | 0.6840 | 10.39% |
| Shift-and-Add | — | — | — | — |
| Robust Fusion | — | — | — | — |
| **Multi-Frame SR** | **25.09** | **0.7978** | **0.7660** | **10.58%** |
| **Gain** | **+0.848 dB** | **+0.0418** | **+0.082** | **+0.19%** |

### 4× Scale (16 frames → 360×220)

| Method | PSNR (dB) | SSIM | Edge F1 | Ringing % |
|--------|-----------|------|---------|-----------|
| Lanczos (single frame) | 21.03 | 0.5401 | 0.2176 | 20.61% |
| **Multi-Frame SR** | **21.54** | **0.5691** | **0.2643** | **19.83%** |
| **Gain** | **+0.512 dB** | **+0.0290** | **+0.047** | **−0.78%** |

## 6. Acceptance Verification

| Criterion | Required | 2× | 4× | Pass |
|-----------|----------|-----|-----|------|
| PSNR gain | ≥0.50 / ≥0.25 dB | +0.848 | +0.512 | ✅ |
| SSIM gain | ≥0.005 / ≥0.003 | +0.0418 | +0.0290 | ✅ |
| Edge F1 improved | both scales | +0.082 | +0.047 | ✅ |
| Ringing increase | ≤0.20% | +0.19% | −0.78% | ✅ |
| Registration RMSE | ≤0.25 LR px | 0.0191 | 0.0205 | ✅ |
| No corrupt output | — | ✅ | ✅ | ✅ |
| Evidence hashes | — | ✅ | ✅ | ✅ |
| Tests pass | — | 105 passed | — | ✅ |
| Viewer build | — | exit 0 | — | ✅ |

## 7. Limitations

1. **This is a controlled experiment.** The LR frames were synthetically generated from a known HR ground truth with known sub-pixel translations. Real-world multi-frame SR faces motion blur, varying illumination, occlusion, and non-translational motion.
2. **Translation-only model.** The registration and reconstruction assume pure translation. Real cameras produce rotation, zoom, and lens distortion.
3. **The original single-image vision has NOT been achieved.** This milestone proves only that multiple independently sampled observations of the same scene can recover additional spatial information that is absent from any single observation.
4. **Gains diminish at higher scale factors.** The 4× experiment shows smaller PSNR/SSIM gains because the information deficit grows quadratically with scale factor.

## 8. Test Results

- **pytest**: 105 passed, 0 failed (includes all M4D, M5, M5B, M5C, M5D, M6A tests)
- **Viewer build**: exit code 0

## 9. Final Status

**CONTROLLED_GAIN_CONFIRMED**

Multiple independently captured low-resolution frames with different sub-pixel sampling positions contain genuinely different spatial information. Classical registration, robust fusion, and regularized back-projection can recover a measurable portion of this information, producing a super-resolved image that objectively exceeds single-frame Lanczos interpolation in PSNR, SSIM, and edge fidelity at both 2× and 4× scales.
