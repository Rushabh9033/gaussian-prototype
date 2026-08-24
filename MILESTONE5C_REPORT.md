# Milestone 5C Closure: Internal Self-Example Patch Reconstruction (ISEPR)

## Objective
The objective of this milestone was to test whether repeated structures and textures already present inside a single photograph can provide useful source-supported high-frequency residuals during enlargement. This strictly mathematical experiment prevents any hallucination by only transferring true pixel differences sourced directly from the input.

## Architecture

This algorithm is based on the general published principle that natural images contain multi-scale recurrence (e.g., "Super-Resolution from a Single Image," Glasner et al., 2009; "Image and Video Upscaling from Local Self-Examples," Freedman and Fattal, 2011).

1. **Deterministic Scale-Space**: Built using strict anti-aliased area downsampling directly from the original source in linear RGB to create scales [1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25].
2. **Cross-Scale Patch Dictionary**: High-frequency residuals are calculated dynamically per scale. The Low surrogate is formed by mimicking the target scale downsampling/upsampling process. Residual = High_Exemplar - Low_Surrogate.
3. **Patch Search**: Exact nearest-neighbor (`cKDTree`) on deterministic luminance descriptors (normalized pixels, mean, standard deviation, gradient mean absolute errors).
4. **Confidence Gating**: Residual transfer is gated by an absolute distance threshold, best-to-second-best distance ratio, and variance matching. Flat areas or mismatched descriptors default gracefully to the Lanczos baseline with 0.0 confidence.
5. **Overlap-Add Reconstruction**: Uses a Hann window to safely accumulate patches across overlapping regions, bounded by 256x256 tiles to guarantee constant low RAM usage. 

## Experimental Configurations

- **ISEPR_A (Conservative)**: Small patches (5x5), Top-K=3. Strict maximum distance gating (0.5), strict distance ratio threshold (0.8), low residual gain (0.5) to prevent over-sharpening.
- **ISEPR_B (Balanced)**: Small patches (5x5), Top-K=5. Balanced distance (1.0), ratio (0.9), and medium residual gain (0.8).
- **ISEPR_C (Texture-seeking)**: Medium patches (7x7), Top-K=7. Permissive distance (1.5), permissive ratio (0.95), maximum residual gain (1.0).

## Results and Limitations

The execution successfully transferred true high-frequency source data to the enlarged images. Because the residuals originate solely from the input image, they inherently contain its noise floor and structure limits. 

The configurations were heavily restricted by the dictionary's actual contents—if a matching sharp edge didn't exist in the input scale-space, the algorithm mathematically could not "invent" one.

*Note: The official benchmark dictates that if structural metrics (PSNR/SSIM) suffer unacceptable mathematical loss, or if Edge F1 fails to definitively improve over Lanczos across scales, no configuration is automatically accepted.*

### Status: `NO_MEASURABLE_GAIN`

While the architectural transfer mechanism functions mathematically, and internal scale-space patches were proven to successfully match and transfer high-frequency energy, the quantitative results did not beat Lanczos cleanly. 

**Visual Analysis**:
- The residuals provided a slight sharpening to distinct edges (like the license plate) when high-confidence matches existed.
- However, self-example residuals often carry noise, meaning transferred patches amplify local grain structure.
- When an exact match fails, confidence falls to 0, leaving "patchy" transitions between sharpened regions and Lanczos baseline regions.

**Conclusion**:
Internal patch recurrence is a mathematically true, non-hallucinated method to add high-frequency energy. But without an external dataset of "perfect" pristine edges, the transferred residuals are bound by the captured blur and noise of the source photograph itself. No missing information was proven recovered.
