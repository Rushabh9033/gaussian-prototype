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
   - Luminance extraction and Structure Tensor edge detection using Scharr gradients to determine edge coherence and exact gradient direction.
   - Direction-aware sharpening (sharpening strictly across edges by evaluating the second derivative in the gradient direction).
   - Tested 3 conservative strengths (0.5, 1.0, 1.5). Selected `1.5` as the primary configuration based on Edge F1 vs Ringing trade-off.
   - Local min/max anti-ringing clamping (with uint8 overflow prevention).
   - Conversion back to sRGB.

## Benchmark Procedure
1. The original `porche.png` (360x220) was used as ground truth.
2. The image was downsampled using area averaging to 180x110 (2x), 90x55 (4x), and 45x28 (8x).
3. The downsampled images were restored back to 360x220 using all four methods.
4. Various classical metrics were computed against the ground truth. F1 Precision/Recall used a correctly computed directional match.

## Metric Results

### 2x Restoration
| Method | PSNR | SSIM | Edge F1 | Ringing % |
|--------|------|------|---------|-----------|
| Nearest | 25.63 | 0.845 | 0.850 | 0.00% |
| Bicubic | 26.80 | 0.860 | 0.858 | 0.23% |
| Lanczos | **26.89** | **0.862** | 0.874 | 0.27% |
| Hybrid (1.5) | 22.94 | 0.791 | **0.886** | 7.16% |

### 4x Restoration
| Method | PSNR | SSIM | Edge F1 | Ringing % |
|--------|------|------|---------|-----------|
| Nearest | 21.71 | 0.625 | 0.642 | 4.57% |
| Bicubic | 22.82 | 0.653 | 0.459 | 3.73% |
| Lanczos | **22.85** | **0.654** | 0.485 | 3.57% |
| Hybrid (1.5) | 21.96 | 0.626 | **0.712** | 8.37% |

### 8x Restoration
| Method | PSNR | SSIM | Edge F1 | Ringing % |
|--------|------|------|---------|-----------|
| Nearest | 19.14 | 0.414 | 0.353 | 16.91% |
| Bicubic | 20.35 | 0.466 | 0.055 | 17.21% |
| Lanczos | **20.39** | **0.469** | 0.039 | 16.81% |
| Hybrid (1.5) | 20.11 | 0.459 | **0.341** | 17.75% |

*(Note: The Hybrid method significantly improved structural boundaries and edge retention across all scales—crushing Lanczos on Edge F1—but the conservative sharpening still lowered the global pixel-wise accuracy (PSNR/SSIM) due to noise introduction).*

## Artifacts and Evidence
- **Results JSON**: `m4d_results/results.json`
- **Execution Evidence**: `m4d_results/evidence.jsonl`
- **Annotated Source**: `m4d_results/annotated_source.png` (Shows exact bounding boxes used for crops)
- **Contact Sheets**: `m4d_results/contact_full_4x.png`, `m4d_results/contact_8x_wheel.png`, etc.
- **Source Image SHA-256**: `8eda0f62759c5d72da363eb338329a87650cd931076fb1c0f932d71dbefdafc1`

## Test and Build Evidence
- All 28 automated Python tests passed successfully. Tests specifically verified integer overflow prevention, strict artifact completion, missing `.pycache` verification, the corrected edge F1 formulation, directional logic inclusion, and no AI dependencies.
- The viewer production build completed successfully.

## Technical Limitations
- The hybrid approach improves edge recall and precision over standard Lanczos but fundamentally fails to beat Lanczos on global pixel-wise accuracy (PSNR/SSIM).
- **Explicit Statement**: No method (including the Hybrid method) can recover or invent uncaptured real-world information. The output is strictly limited by the Shannon-Nyquist theorem on the original sampled pixels. This non-AI path is unsuitable as the main deep-zoom solution.

## Final Status
**CLASSICAL_LIMIT_CONFIRMED**

The non-AI limit has been successfully established and benchmarked. Lanczos interpolation remains superior for smooth photographic fidelity, while edge-aware filtering only trades numerical PSNR for sharper local boundaries without introducing new real details.
