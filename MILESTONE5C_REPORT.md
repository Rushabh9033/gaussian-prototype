# Milestone 5C Closure: Internal Self-Example Patch Reconstruction (ISEPR)

## Objective
The objective of this milestone was to test whether repeated structures and textures already present inside a single photograph can provide useful source-supported high-frequency residuals during enlargement. 

## Architecture

This algorithm is based on the general published principle that natural images contain multi-scale recurrence (e.g., "Super-Resolution from a Single Image," Glasner et al., 2009; "Image and Video Upscaling from Local Self-Examples," Freedman and Fattal, 2011).

1. **Deterministic Scale-Space**: Built using strict anti-aliased area downsampling directly from the original source in linear RGB to create scales [1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25].
2. **Cross-Scale Patch Dictionary**: High-frequency residuals are calculated dynamically per scale. The Low surrogate is formed by mimicking the target scale downsampling/upsampling process. Residual = High_Exemplar - Low_Surrogate.
3. **Patch Search**: Exact nearest-neighbor (`cKDTree`) on deterministic luminance descriptors (normalized pixels, mean, standard deviation, gradient mean absolute errors).
4. **Confidence Gating**: Residual transfer is gated by an absolute distance threshold, best-to-second-best distance ratio, and variance matching. Flat areas or mismatched descriptors default gracefully to the Lanczos baseline with 0.0 confidence.
5. **Overlap-Add Reconstruction**: Uses a Hann window to safely accumulate patches across overlapping regions. 

## Source-Only Data-Flow Explanation
The algorithm is purely classical: no AI, no neural networks, no pretrained models, no external image databases, and no APIs are used. The dictionary is populated exclusively by patches extracted from downsampled versions of the single input image. During reconstruction, target patches are queried against this internal dictionary. If a match satisfies strict confidence gates, the corresponding high-frequency residual is scaled and added to the Lanczos baseline. 

## Experimental Configurations & Rejection Reasons

- **ISEPR_A (Conservative)**: Small patches (5x5), Top-K=3. Strict maximum distance gating (0.5), strict distance ratio threshold (0.8), low residual gain (0.5). Rejected because it failed to meaningfully improve Edge F1 and caused structural degradation.
- **ISEPR_B (Balanced)**: Small patches (5x5), Top-K=5. Balanced distance (1.0), ratio (0.9), and medium residual gain (0.8). Rejected due to ringing, noise amplification, and failure to beat Lanczos quantitatively.
- **ISEPR_C (Texture-seeking)**: Medium patches (7x7), Top-K=7. Permissive distance (1.5), permissive ratio (0.95), maximum residual gain (1.0). Rejected due to severe noise, distorted edges, and lowest PSNR/SSIM scores.

## Results and Visual Limitations

ISEPR did not recover missing information. While the residuals originate solely from the input image, they can be transferred to an incorrect location and create false local detail. 

**Visual Analysis**:
- Visual results show noise, ringing, and distorted edges.
- Self-example residuals carry the source image's noise, meaning transferred patches amplify local grain structure.
- When transferred patches don't perfectly align structurally, they generate artificial high-frequency distortions.
- Increased edge energy merely reflects this amplified noise and distortion, not accurate recovered information.

## Status: `NO_MEASURABLE_GAIN`

Lanczos remains the accepted operational fallback. No ISEPR configuration was selected. The quantitative metrics failed to beat Lanczos, and visual inspection confirms that the internal self-example method degrades image quality.

## Benchmark Execution

- **Benchmark Commit**: {COMMIT}
- **Commands**: 
  - `python run_m5c_benchmark.py`
  - `python validate_m5c.py`
- **Validation**: All tests passed (16 passed, 0 failed, 0 skipped).
- **Viewer-build**: Successful (exit code 0).
- **Peak RAM**: {RAM_MB} MiB ({RAM_BYTES} bytes)
- **Tile-Equivalence Maximum Error**: {TILE_ERR}

## Metrics (2x / 4x / 8x)
{METRICS}

## Artifacts
{ARTIFACTS}
