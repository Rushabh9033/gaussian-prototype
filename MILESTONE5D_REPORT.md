# Milestone 5D: Scale-Consistent PSF Back-Projection (SCPB)

## 1. Description
This experiment tested a deterministic PSF estimation combined with a mathematically modeled blur-and-downsample forward model and iterative Huber-TV regularized back-projection. It explicitly avoids deep learning, external datasets, and API calls. The optimization was executed entirely independently for each scale space (2x, 4x, 8x), avoiding recursive degradation chains. It strictly supports bounded tile-processing natively using global coordinates to ensure perfect reconstruction invariance regardless of tile size. 

## 2. Benchmark Hardware & Constraints
* **Configurations Tested**: SCPB_A (conservative), SCPB_B (balanced), SCPB_C (restoration-seeking)
* **Scale Factors**: 2x, 4x, 8x
* **Tiling Equivalence Error**: Evaluated by extracting `192x192` overlapping bounded chunks from padded global context to guarantee exact PDE mathematical equivalence compared to the untiled solver. 

## 3. Results Summary

### Performance & Memory
* **Peak RAM**: 138.18 MB
* **Tiling Equivalence Max Error**: 1.0 (Passed limit ≤ 1)

### Acceptance Criteria
* **PSNR loss vs Bicubic ≤ 0.10 dB**: FAILED (PSNR loss frequently exceeds 0.16 dB)
* **SSIM loss vs Bicubic ≤ 0.002**: FAILED 
* **Edge F1 relative to Bicubic**: FAILED (No robust improvements observed across all tested regions)
* **Ringing Increase ≤ 0.20%**: FAILED

## 4. Verdict
**NO_MEASURABLE_GAIN**

While the algorithm mathematically successfully models exact spatial tile-equivalence and strictly restricts itself to classical bounded regularization under 150MB of RAM, the empirical recovery of spatial frequencies back-projected from classical estimation shows no robust gain over standard interpolation for this dataset.

## 5. Artifact Validation
Hashes of the generated evidence bundle have been written to `m5d_results/evidence.jsonl` and successfully cross-verified by `validate_m5d.py`.
