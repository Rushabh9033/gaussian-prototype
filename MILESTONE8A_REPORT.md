# Milestone 8A: Local Real-ESRGAN Visual Gate

## Architecture

The milestone evaluates local inference of `RealESRGAN_x4plus` (fp16, tile=0) against Lanczos interpolation on a deterministic 360×220 evaluation image (`m4d_test_assets/porche.png`).

The experiment runs in an isolated Python workspace (`.m8a_workspace/`) using `--system-site-packages` to inherit the system's PyTorch 2.6.0+cu124 installation. The Real-ESRGAN model is orchestrated using `basicsr.archs.rrdbnet_arch.RRDBNet` and `realesrgan.RealESRGANer` from cloned repositories of `xinntao/BasicSR` and `xinntao/Real-ESRGAN`.

Input images are downscaled to 2X (180×110) and 4X (90×55) via OpenCV `INTER_AREA`, then restored to 360×220 using Real-ESRGAN and Lanczos. A third run upscales the original 360×220 to 1440×880 (outscale=4).

## Exact Experiment

```
python run_m8a_benchmark.py
```

This command:
1. Creates `.m8a_workspace/venv` with `--system-site-packages`
2. Clones `xinntao/BasicSR` and `xinntao/Real-ESRGAN`
3. Downloads `RealESRGAN_x4plus.pth` (67 MB)
4. Generates LR-2X and LR-4X inputs from `porche.png`
5. Runs Real-ESRGAN inference (CUDA, fp16, tile=0) for 2X, 4X, and orig scales
6. Computes Lanczos baselines
7. Calculates PSNR, SSIM, Edge-F1, ORB, gradient energy, Laplacian variance, colour histogram distance, pHash, and regional SSIM for plate/headlamp/wheel/silhouette crops
8. Records VRAM telemetry inside the PyTorch process
9. Writes `execution_evidence.json` and output images

## Metrics

### 2X Upscale (180×110 → 360×220)

| Metric | Lanczos | Real-ESRGAN | Change | Guardrail | Result |
|--------|---------|-------------|--------|-----------|--------|
| SSIM | 0.8500 | 0.7500 | −0.1000 | ≤0.01 deg | **FAIL** |
| PSNR (dB) | 26.84 | 23.69 | −3.15 | ≤1.0 deg | **FAIL** |
| Edge-F1 | 0.5736 | 0.5784 | **+0.0048** | ≤0.02 deg | PASS |
| ORB Inlier Ratio | 0.9984 | 0.9897 | −0.0087 | ≤0.05 deg | PASS |
| Plate SSIM | 0.7958 | 0.6411 | −0.1547 | ≤0.03 deg | **FAIL** |
| Headlamp SSIM | 0.8658 | 0.7846 | −0.0812 | ≤0.03 deg | **FAIL** |
| Wheel SSIM | 0.8253 | 0.7360 | −0.0892 | ≤0.03 deg | **FAIL** |
| Silhouette SSIM | 0.8496 | 0.7470 | −0.1026 | ≤0.03 deg | **FAIL** |

### 4X Upscale (90×55 → 360×220)

| Metric | Lanczos | Real-ESRGAN | Change | Guardrail | Result |
|--------|---------|-------------|--------|-----------|--------|
| SSIM | 0.6301 | 0.5928 | −0.0373 | ≤0.01 deg | **FAIL** |
| PSNR (dB) | 22.84 | 21.18 | −1.66 | ≤1.0 deg | **FAIL** |
| Edge-F1 | 0.2448 | 0.3241 | **+0.0793** | ≤0.02 deg | PASS |
| ORB Inlier Ratio | 0.9486 | 0.9730 | +0.0244 | ≤0.05 deg | PASS |
| Plate SSIM | 0.4985 | 0.4053 | −0.0932 | ≤0.03 deg | **FAIL** |
| Headlamp SSIM | 0.5557 | 0.4870 | −0.0687 | ≤0.03 deg | **FAIL** |
| Wheel SSIM | 0.5392 | 0.4970 | −0.0421 | ≤0.03 deg | **FAIL** |
| Silhouette SSIM | 0.6210 | 0.5741 | −0.0469 | ≤0.03 deg | **FAIL** |

### Edge-F1 Interpretation

Real-ESRGAN improved Edge-F1 at both scales: +0.0048 at 2X and +0.0793 at 4X. This reflects the model's ability to synthesize sharp, high-contrast edges. However, these "improved" edges represent hallucinated structures — not recovered factual detail. The generated edges (e.g., garbled license plate characters, fabricated wheel spoke patterns) do not match the ground-truth geometry. These edge improvements do not overcome the catastrophic SSIM, PSNR, colour-histogram, and regional-fidelity failures documented above.

## Runtime and VRAM

| Run | Time (s) | Peak Allocated (GB) | Peak Reserved (GB) |
|-----|----------|--------------------|--------------------|
| 2X (180×110→360×220) | 0.550 | 0.195 | 0.277 |
| 4X (90×55→360×220) | 0.082 | 0.195 | 0.277 |
| Orig (360×220→1440×880) | 0.340 | 0.684 | 0.885 |

Maximum peak reserved VRAM: **0.885 GB** — well within the 7.5 GB limit.

## Source and Model Provenance

| Item | Value |
|------|-------|
| Real-ESRGAN commit | `a4abfb2979a7bbff3f69f58f58ae324608821e27` |
| BasicSR commit | `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Python | 3.13.7 |
| OS | Windows 11 |

## Artifact Table

| Artifact | SHA-256 |
|----------|---------|
| source/porche.png | `8eda0f62759c5d72da363eb338329a87650cd931076fb1c0f932d71dbefdafc1` |
| controlled_inputs/lr_2x_180x110.png | `6823545765553714bb41e2aed79f71d0a5959b8cf58b88a28872d7ba4d239316` |
| controlled_inputs/lr_4x_90x55.png | `c5abfce38d105180f517abda72316f782c192341a274aa7f2f54a40da0a8b77b` |
| realesrgan/2x_360x220.png | `52b0ad8c843b28913b8646172fdac070e2dbc846f04329483488f9fd44ab7710` |
| realesrgan/4x_360x220.png | `4898c30ae19a7765b5bc75ed4351970c3a42ee9712675f1d93661f8bb776505f` |
| realesrgan/orig_1440x880.png | `da43f532174022c83b3f0dff653b4a45dae734c938788630531cc907b41dd99d` |

## Test Results

```
python -m pytest -q tests/test_m8a.py
```

31 tests passed covering: deterministic inputs, PNG dimensions, execution evidence schema, SHA-256 hash verification, results.json schema, closure verification schema, crop presence and Git tracking, report content, contact sheet non-blankness, negative-result validation, no-model-download safety, no-API-call safety, and viewer/V3 non-modification.

```
python validate_m8a.py
```

Output: `VALID: experiment failed quality guardrails and status is correctly NO_USEFUL_GAIN`

## Viewer Build

```
cd src/viewer && npm ci && npm run build
```

Build succeeded. No viewer or V3 format modifications were made during this milestone.

## Visual Limitations

- Real-ESRGAN hallucinates plausible but incorrect detail in every region tested.
- The license plate region is particularly degraded: SSIM drops from 0.796 to 0.641 (2X) and from 0.498 to 0.405 (4X).
- Colour histogram distance worsened at both scales (0.041 → 0.123 at 2X, 0.094 → 0.132 at 4X).
- The model cannot recover factual text or geometric structures that were lost during downscaling.

## Licensing

- **BasicSR** and **Real-ESRGAN** source code: BSD-3-Clause license.
- **RealESRGAN_x4plus model weights**: distributed under academic/research terms. A commercial-license review is required before any production deployment.

## Final Status

**`NO_USEFUL_GAIN`**

Real-ESRGAN provides no useful structural improvement over Lanczos for rigorous scientific multi-frame super-resolution. All primary quality gates (SSIM, PSNR, regional SSIM) failed catastrophically at both scales. The slight Edge-F1 improvements reflect hallucinated sharpness, not recovered ground-truth detail.
