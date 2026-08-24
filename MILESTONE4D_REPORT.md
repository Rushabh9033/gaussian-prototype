# Milestone 4D: Non-AI Classical Zoom Quality Benchmark

## Architecture & Dependencies
This milestone evaluates the maximum faithful zoom quality achievable without AI models.

**Dependencies:**
- Python 3.11
- NumPy
- Pillow
- OpenCV (cv2) - excluding DNN module
- scikit-image
- SciPy

**Non-AI Architecture Tested:**
1. **Nearest-neighbour**: Basic cv2.INTER_NEAREST.
2. **Bicubic**: Basic cv2.INTER_CUBIC.
3. **Lanczos**: PIL Image.Resampling.LANCZOS.
4. **Classical Edge-Aware Hybrid**: 
   - Conversion from sRGB to Linear RGB
   - Lanczos upscale as the base
   - Luminance extraction and Structure Tensor edge detection using Scharr gradients to determine edge coherence.
   - Conservative unsharp mask sharpening modulated by edge coherence.
   - Local min/max anti-ringing clamping.
   - Conversion back to sRGB.

## Benchmark Procedure
1. The original `porche.png` (360x220) was used as ground truth.
2. The image was downsampled using area averaging to 180x110 (2x), 90x55 (4x), and 45x28 (8x).
3. The downsampled images were restored back to 360x220 using all four methods.
4. Various classical metrics were computed against the ground truth.

## Metric Results

### 2x Restoration
| Method | PSNR | SSIM | Edge F1 | Ringing % |
|--------|------|------|---------|-----------|
| Nearest | 25.63 | 0.845 | 0.831 | 5.18% |
| Bicubic | 26.80 | 0.860 | 0.852 | 5.41% |
| Lanczos | **26.89** | **0.862** | 0.869 | 5.45% |
| Hybrid | 25.16 | 0.839 | **0.909** | 9.57% |

### 4x Restoration
| Method | PSNR | SSIM | Edge F1 | Ringing % |
|--------|------|------|---------|-----------|
| Nearest | 21.71 | 0.625 | **0.627** | 9.46% |
| Bicubic | 22.82 | 0.653 | 0.439 | 8.66% |
| Lanczos | **22.85** | **0.654** | 0.465 | 8.53% |
| Hybrid | 22.36 | 0.634 | 0.581 | 12.47% |

### 8x Restoration
| Method | PSNR | SSIM | Edge F1 | Ringing % |
|--------|------|------|---------|-----------|
| Nearest | 19.14 | 0.414 | **0.347** | 20.94% |
| Bicubic | 20.35 | 0.466 | 0.048 | 21.26% |
| Lanczos | **20.39** | **0.469** | 0.036 | 20.86% |
| Hybrid | 20.00 | 0.449 | 0.132 | 23.56% |

*(Note: Nearest neighbor retains artificially high Edge F1 due to strong pixelated block boundaries being detected as edges, whereas smooth interpolations lose edges entirely at extreme downsampling. The Hybrid method successfully restored more edges than Lanczos/Bicubic at all scales.)*

## Artifacts and Evidence
- **Results JSON**: `m4d_results/results.json`
- **Execution Evidence**: `m4d_results/evidence.jsonl`
- **Contact Sheets**: `m4d_results/contact_full_4x.png`, `m4d_results/contact_8x_wheel.png`, etc.
- **Source Image SHA-256**: `8eda0f62759c5d72da363eb338329a87650cd931076fb1c0f932d71dbefdafc1`

## Test and Build Evidence
- All 28 automated Python tests passed successfully. Tests verified deterministic output, RGB bounds, absence of AI dependencies, correct output dimensions, and metric correctness.
- The viewer production build (`npm run build`) completed successfully with 0 vulnerabilities.

## Technical Limitations
- The hybrid approach improves edge recall and precision (F1) over standard Lanczos but fundamentally fails to beat Lanczos on global pixel-wise accuracy (PSNR/SSIM) due to the noise and ringing introduced by sharpening.
- **Explicit Statement**: No method (including the Hybrid method) can recover or invent uncaptured real-world information. The output is strictly limited by the Shannon-Nyquist theorem on the original sampled pixels.

## Final Status
**CLASSICAL_LIMIT_CONFIRMED**

The experiment ran correctly and successfully created the hybrid method which improved edge retention. However, Lanczos remains mathematically superior on SSIM and PSNR. This confirms the limits of non-AI classical upscaling for this repository.
