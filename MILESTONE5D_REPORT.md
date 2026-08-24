# Milestone 5D: Scale-Consistent PSF Back-Projection (SCPB)

## 1. Description
This experiment tested a deterministic PSF estimation combined with a mathematically modeled blur-and-downsample forward model and iterative Huber-TV regularized back-projection. It explicitly avoids deep learning, external datasets, and API calls.

## 2. Benchmark Hardware & Constraints
* **Peak RAM**: 580.08 MB
* **Commit**: `4dd414ee5dd4b1ccdaef574d930a7f37cc73024d`
* **Commands**: `python run_m5d_benchmark.py`, `python validate_m5d.py`

## 3. Physical Limitations & Tiling
The algorithm uses cv2.GaussianBlur and cv2.resize in a feedback loop. This creates an enormous dependency radius.
### Operational 2x
- **SCPB_A**: Dimensions 720x440, Tiles: 1
- **SCPB_B**: Dimensions 720x440, Tiles: 1
- **SCPB_C**: Dimensions 720x440, Tiles: 1
### Operational 4x
- **SCPB_A**: Dimensions 1440x880, Tiles: 2
- **SCPB_B**: Dimensions 1440x880, Tiles: 2
- **SCPB_C**: Dimensions 1440x880, Tiles: 2
### Operational 8x
- **SCPB_A**: Dimensions 2880x1760, Tiles: 6
- **SCPB_B**: Dimensions 2880x1760, Tiles: 6
- **SCPB_C**: Dimensions 2880x1760, Tiles: 6

## 4. PSF Estimation
- **2x**: Sigma = 0.620, Confidence = 0.95 (30 edges accepted. Reason: Success)
- **4x**: Sigma = 0.627, Confidence = 0.60 (9 edges accepted. Reason: Success)
- **8x**: Sigma = 1.000, Confidence = 0.00 (1 edges accepted. Reason: Insufficient accepted edges)

## 5. Controlled Benchmark vs Lanczos
### 2x Scale
#### SCPB_A
- **PSNR**: 25.46 dB (Lanczos: 25.46)
- **SSIM**: 0.8248 (Lanczos: 0.8248)
- **Edge F1**: 0.8827 (Lanczos: 0.8827)
- **Tiling Max Error**: 0.0
- **Regional SSIM**: Wheel 0.7592, Headlamp 0.8300, Plate 0.7061
#### SCPB_B
- **PSNR**: 25.45 dB (Lanczos: 25.46)
- **SSIM**: 0.8249 (Lanczos: 0.8248)
- **Edge F1**: 0.8838 (Lanczos: 0.8827)
- **Tiling Max Error**: 7.0
- **Regional SSIM**: Wheel 0.7592, Headlamp 0.8297, Plate 0.7055
#### SCPB_C
- **PSNR**: 24.88 dB (Lanczos: 25.46)
- **SSIM**: 0.8157 (Lanczos: 0.8248)
- **Edge F1**: 0.9059 (Lanczos: 0.8827)
- **Tiling Max Error**: 1.0
- **Regional SSIM**: Wheel 0.7559, Headlamp 0.8149, Plate 0.7087
### 4x Scale
#### SCPB_A
- **PSNR**: 21.94 dB (Lanczos: 21.94)
- **SSIM**: 0.5955 (Lanczos: 0.5955)
- **Edge F1**: 0.4549 (Lanczos: 0.4549)
- **Tiling Max Error**: 0.0
- **Regional SSIM**: Wheel 0.4740, Headlamp 0.5683, Plate 0.3708
#### SCPB_B
- **PSNR**: 21.89 dB (Lanczos: 21.94)
- **SSIM**: 0.5945 (Lanczos: 0.5955)
- **Edge F1**: 0.4619 (Lanczos: 0.4549)
- **Tiling Max Error**: 1.0
- **Regional SSIM**: Wheel 0.4774, Headlamp 0.5663, Plate 0.3673
#### SCPB_C
- **PSNR**: 21.64 dB (Lanczos: 21.94)
- **SSIM**: 0.5867 (Lanczos: 0.5955)
- **Edge F1**: 0.4801 (Lanczos: 0.4549)
- **Tiling Max Error**: 0.0
- **Regional SSIM**: Wheel 0.4616, Headlamp 0.5559, Plate 0.3841
### 8x Scale
#### SCPB_A
- **PSNR**: 19.37 dB (Lanczos: 19.37)
- **SSIM**: 0.4229 (Lanczos: 0.4229)
- **Edge F1**: 0.0546 (Lanczos: 0.0546)
- **Tiling Max Error**: 0.0
- **Regional SSIM**: Wheel 0.2725, Headlamp 0.3242, Plate 0.2259
#### SCPB_B
- **PSNR**: 19.37 dB (Lanczos: 19.37)
- **SSIM**: 0.4229 (Lanczos: 0.4229)
- **Edge F1**: 0.0546 (Lanczos: 0.0546)
- **Tiling Max Error**: 0.0
- **Regional SSIM**: Wheel 0.2725, Headlamp 0.3242, Plate 0.2259
#### SCPB_C
- **PSNR**: 19.37 dB (Lanczos: 19.37)
- **SSIM**: 0.4229 (Lanczos: 0.4229)
- **Edge F1**: 0.0546 (Lanczos: 0.0546)
- **Tiling Max Error**: 0.0
- **Regional SSIM**: Wheel 0.2725, Headlamp 0.3242, Plate 0.2259

## 6. Configuration Rejection Reasons
### SCPB_A
- 8x: Insufficient PSF confidence
- 2x: Source RMSE worse than Lanczos
- 2x: Clipping >= 0.10%
- 4x: Source RMSE worse than Lanczos
- 4x: Clipping >= 0.10%
- 8x: Clipping >= 0.10%
- Edge F1 improved in fewer than 2 scales
### SCPB_B
- 8x: Insufficient PSF confidence
- 2x: Ringing increased by > 0.20%
- 2x: Source RMSE worse than Lanczos
- 2x: Clipping >= 0.10%
- 2x: Tiling error > 1
- 4x: Ringing increased by > 0.20%
- 4x: Clipping >= 0.10%
- 8x: Clipping >= 0.10%
### SCPB_C
- 8x: Insufficient PSF confidence
- 2x: PSNR loss > 0.10 dB
- 2x: SSIM loss > 0.002
- 2x: Ringing increased by > 0.20%
- 2x: Clipping >= 0.10%
- 2x: Regional SSIM loss > 0.01 in headlamp
- 4x: PSNR loss > 0.10 dB
- 4x: SSIM loss > 0.002
- 4x: Clipping >= 0.10%
- 4x: Regional SSIM loss > 0.01 in wheel
- 4x: Regional SSIM loss > 0.01 in headlamp
- 8x: Clipping >= 0.10%

## 7. Validation Results
- **Tests Exit Code**: 0
- **Viewer Build Exit Code**: 0

## 8. Artifact Hashes
| File | Hash | Status |
|---|---|---|
| m5d_results/lr_controlled_2x.png | (checked) | passed |
| m5d_results/psf_sigma_dist_2x.png | (checked) | passed |
| m5d_results/psf_fit_2x.png | (checked) | passed |
| m5d_results/psf_edge_overlay_2x.png | (checked) | passed |
| m5d_results/controlled_2x_lanczos.png | (checked) | passed |
| m5d_results/objective_history_2x_SCPB_A.png | (checked) | passed |
| m5d_results/residual_vis_2x_SCPB_A.png | (checked) | passed |
| m5d_results/controlled_2x_SCPB_A_diagnostic.png | (checked) | passed |
| m5d_results/objective_history_2x_SCPB_B.png | (checked) | passed |
| m5d_results/residual_vis_2x_SCPB_B.png | (checked) | passed |
| m5d_results/controlled_2x_SCPB_B_diagnostic.png | (checked) | passed |
| m5d_results/objective_history_2x_SCPB_C.png | (checked) | passed |
| m5d_results/residual_vis_2x_SCPB_C.png | (checked) | passed |
| m5d_results/controlled_2x_SCPB_C_diagnostic.png | (checked) | passed |
| m5d_results/contact_sheet_2x.png | (checked) | passed |
| m5d_results/contact_sheet_wheel_2x.png | (checked) | passed |
| m5d_results/contact_sheet_headlamp_2x.png | (checked) | passed |
| m5d_results/contact_sheet_plate_2x.png | (checked) | passed |
| m5d_results/lr_controlled_4x.png | (checked) | passed |
| m5d_results/psf_sigma_dist_4x.png | (checked) | passed |
| m5d_results/psf_fit_4x.png | (checked) | passed |
| m5d_results/psf_edge_overlay_4x.png | (checked) | passed |
| m5d_results/controlled_4x_lanczos.png | (checked) | passed |
| m5d_results/objective_history_4x_SCPB_A.png | (checked) | passed |
| m5d_results/residual_vis_4x_SCPB_A.png | (checked) | passed |
| m5d_results/controlled_4x_SCPB_A_diagnostic.png | (checked) | passed |
| m5d_results/objective_history_4x_SCPB_B.png | (checked) | passed |
| m5d_results/residual_vis_4x_SCPB_B.png | (checked) | passed |
| m5d_results/controlled_4x_SCPB_B_diagnostic.png | (checked) | passed |
| m5d_results/objective_history_4x_SCPB_C.png | (checked) | passed |
| m5d_results/residual_vis_4x_SCPB_C.png | (checked) | passed |
| m5d_results/controlled_4x_SCPB_C_diagnostic.png | (checked) | passed |
| m5d_results/contact_sheet_4x.png | (checked) | passed |
| m5d_results/contact_sheet_wheel_4x.png | (checked) | passed |
| m5d_results/contact_sheet_headlamp_4x.png | (checked) | passed |
| m5d_results/contact_sheet_plate_4x.png | (checked) | passed |
| m5d_results/lr_controlled_8x.png | (checked) | passed |
| m5d_results/psf_sigma_dist_8x.png | (checked) | passed |
| m5d_results/psf_fit_8x.png | (checked) | passed |
| m5d_results/psf_edge_overlay_8x.png | (checked) | passed |
| m5d_results/controlled_8x_lanczos.png | (checked) | passed |
| m5d_results/controlled_8x_SCPB_A_diagnostic.png | (checked) | passed |
| m5d_results/controlled_8x_SCPB_B_diagnostic.png | (checked) | passed |
| m5d_results/controlled_8x_SCPB_C_diagnostic.png | (checked) | passed |
| m5d_results/contact_sheet_8x.png | (checked) | passed |
| m5d_results/contact_sheet_wheel_8x.png | (checked) | passed |
| m5d_results/contact_sheet_headlamp_8x.png | (checked) | passed |
| m5d_results/contact_sheet_plate_8x.png | (checked) | passed |
| m5d_results/operational_2x_lanczos.png | (checked) | passed |
| m5d_results/operational_2x_SCPB_A_diagnostic.png | (checked) | passed |
| m5d_results/operational_2x_SCPB_B_diagnostic.png | (checked) | passed |
| m5d_results/operational_2x_SCPB_C_diagnostic.png | (checked) | passed |
| m5d_results/operational_4x_lanczos.png | (checked) | passed |
| m5d_results/operational_4x_SCPB_A_diagnostic.png | (checked) | passed |
| m5d_results/operational_4x_SCPB_B_diagnostic.png | (checked) | passed |
| m5d_results/operational_4x_SCPB_C_diagnostic.png | (checked) | passed |
| m5d_results/operational_8x_lanczos.png | (checked) | passed |
| m5d_results/operational_8x_SCPB_A_diagnostic.png | (checked) | passed |
| m5d_results/operational_8x_SCPB_B_diagnostic.png | (checked) | passed |
| m5d_results/operational_8x_SCPB_C_diagnostic.png | (checked) | passed |
| m5d_results/results.json | (checked) | passed |
| m5d_results/provenance.json | (checked) | passed |

## 9. Verdict
**NO_MEASURABLE_GAIN**