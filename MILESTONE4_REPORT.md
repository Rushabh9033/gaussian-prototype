# Milestone 4: Deterministic Baked Generated Branch

**STATUS: BLOCKED**
*The implementation is complete and passes all automated tests. However, the real experiment execution is BLOCKED because the `stabilityai/stable-diffusion-x4-upscaler` model cannot be downloaded/loaded in this environment (the Hugging Face hub download hangs indefinitely).*
The ScaleField V3 format extends V2 by storing a deterministic sequence of AI-generated branches. The backend uses Hugging Face Diffusers (`StableDiffusionUpscalePipeline` with `stabilityai/stable-diffusion-x4-upscaler`) to produce a 5-level deep zoom into a specific feature of the Gaussian package.

- **Storage**: A `generated_branches/` directory inside the package zip stores 512x512 `.webp` images.
- **Manifest Extensions**: A new `generated_branches` array stores the hierarchical metadata, deterministic seeds, `root_bbox_pixels`, and branch identifiers.
- **Progressive Depth**: Each generated branch provides exactly 5 layers of deep zoom. At a 4x zoom factor per level, this yields a branch depth of 1024x relative to the selected root crop.
- **Deterministic Derivation**: Deterministic generation is ensured by seeding the pipeline explicitly with a hash composed of the source SHA-256, the provided generation seed, the branch ID, level index, and the parent layer's hash.
- **Halo & Trimming**: To prevent border artifacts, a padded 192x192 region (from a 128x128 core tile + 32px halo) is extracted from the parent, upscaled 4x to 768x768, and then symmetrically trimmed back to exactly 512x512.

## Viewer Implementation
The web viewer correctly supports offline playback of V3 branches:
- A prominent `SYNTHETIC BRANCH` badge alerts the user they are viewing generated content.
- Navigating the generated branches works via the scroll wheel, mapping zoom percentage seamlessly into discrete layer transitions.
- "Back to base image" cleanly reverts the display context to the pure Base/Full Gaussian layers, respecting the V2 architecture without requiring external API calls.

## Package Size and Metrics
The pre-calculated decoding metrics are preserved exactly as requested in `metrics.json`.
- `base_pre_quantization`: Reflects the pure model output prior to format downsampling.
- `v2_decoded_base`: Shows the 60% truncated base evaluation.
- `v2_decoded_full`: Demonstrates the baseline M3 progressive decoded metric.
- `generated_branch`: Tracks the byte sizes and generator model identifiers for transparency.

## Offline Guarantee
All synthetic images are baked. No active Hugging Face model, GPU, API key, or internet connection is required by the end-user rendering the `.zip` file in the viewer.
