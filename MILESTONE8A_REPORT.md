# Milestone 8A: Local Real-ESRGAN Visual Gate

## Evaluation Gate Results
The milestone evaluated the local inference of `RealESRGAN_x4plus` (fp16, tile=0) on a deterministic 360x220 evaluation image ("porche.png") downscaled to 2X (180x110) and 4X (90x55), comparing the results to standard Lanczos interpolation.

The evaluation clearly indicates that Real-ESRGAN hallucinates visually pleasing but statistically inaccurate textures, causing severe degradation in structural fidelity metrics.

**2X Upscale Evaluation (180x110 -> 360x220):**
- **SSIM:** Degraded by **0.100** (Lanczos: 0.850, Real-ESRGAN: 0.750) — *Failed Guardrail (Limit 0.01)*
- **PSNR:** Degraded by **3.15 dB** (Lanczos: 26.84 dB, Real-ESRGAN: 23.69 dB) — *Failed Guardrail (Limit 1.0 dB)*
- **Edge F1:** Degraded by **0.015** — *Pass*
- **Regional Plate SSIM:** Degraded by **0.154** (Lanczos: 0.795, Real-ESRGAN: 0.641) — *Failed Guardrail (Limit 0.03)*

**4X Upscale Evaluation (90x55 -> 360x220):**
- **SSIM:** Degraded by **0.037** (Lanczos: 0.630, Real-ESRGAN: 0.592) — *Failed Guardrail*
- **PSNR:** Degraded by **1.65 dB** (Lanczos: 22.84 dB, Real-ESRGAN: 21.18 dB) — *Failed Guardrail*
- **Regional Plate SSIM:** Degraded by **0.093** (Lanczos: 0.498, Real-ESRGAN: 0.405) — *Failed Guardrail*

## Hardware Telemetry
The inference was executed using the official `xinntao/Real-ESRGAN` model running locally on PyTorch with CUDA support.

- **OS Environment:** Windows (Python 3.13)
- **Execution Target:** CUDA (RTX 4060 laptop GPU limits)
- **Parameters:** `tile=0`, `half=True`, `outscale=4`
- **Peak VRAM Allocated:** 0.68 GB (during `orig` 1440x880 upscale)
- **Peak VRAM Reserved:** 0.88 GB (during `orig` 1440x880 upscale)
- **Inference Time (360x220 -> 1440x880):** 0.34 seconds

Hardware requirements were easily met, staying well below the 7.5 GB ceiling. 

## Validation Conclusion
While Real-ESRGAN easily executes within our laptop hardware budget, it fundamentally fails the statistical fidelity guardrails. The model introduces widespread generative hallucinations (particularly catastrophic in the license plate region where SSIM drops massively from 0.795 to 0.641). Because it destroys ground-truth geometric structures rather than accurately reconstructing them, it provides no structural advantage over Lanczos for rigorous scientific multi-frame analysis.

**Final Status:** `NO_USEFUL_GAIN`
