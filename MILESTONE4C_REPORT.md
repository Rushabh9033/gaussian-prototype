# MILESTONE 4C: SCALEFIELD-LITE DIRECT-SOURCE PROTOTYPE

**GOAL:** Find whether a fundamentally different architecture can generate five coherent, sharp, synthetic deep-zoom levels from one normal photograph.

## ARCHITECTURE
This milestone implements **ScaleField-Lite**, an original lightweight direct-from-source zoom generator. It abandons recursive image-to-image chains in favor of direct source conditioning and frequency anchoring. 

### Original Components
1. **Source-Coordinate Zoom Path:** Crop boxes are mapped directly to the original input photograph (1x, 4x, 16x, 64x, 256x). 
2. **Direct Source Conditioning:** The model never sees its own generated output from the previous scale as an input. It receives a LANCZOS-resampled crop of the original photo.
3. **Deterministic Scale Identity:** Seeds are generated from `SHA256(source) + bbox + zoom + version`.
4. **Cross-Scale Frequency Anchoring:** A post-generation deterministic correction. The newly generated high-frequency detail is preserved, while the low-frequency color gradients are anchored to the previous scale's generated output to ensure global consistency across zooms.

### Foundation Model Disclosure
* **Model:** `stabilityai/sd-turbo`
* **Pipeline:** `AutoPipelineForImage2Image` (Diffusers)
* **License Warning:** SD-Turbo is provided for non-commercial research purposes under the Stability AI Non-Commercial Research Community License. It must be reviewed before commercial distribution.

## HARDWARE ENVIRONMENT
* GPU: NVIDIA GeForce RTX 4060 Laptop GPU
* VRAM: 8.59 GB
* Framework: PyTorch 2.6.0+cu124 (Python 3.11 Isolated Environment)

## MEASUREMENTS
* **Actual Peak VRAM (Allocated):** [Waiting for execution]
* **Actual Peak VRAM (Reserved):** [Waiting for execution]
* **Model Revision:** b261bac6fd2cf515557d5d0707481eafa0485ec2

## ARTIFACTS
| Artifact | SHA-256 |
| -------- | ------- |
| (populated post-run) | |

## TESTS & VIEWER BUILD
* Python tests: `tests/test_m4c.py` (Passed)
* All existing tests passed
* Viewer build successful (no AI required)

## LIMITATIONS
* At extreme zooms (256x), the source crop is sub-pixel (e.g., 1.4x0.8 pixels). The model must hallucinate all structure, heavily relying on the text prompt and frequency anchoring to remain coherent.
* The absence of recursive structural control means the 256x image might not perfectly align its features with the 64x image if the prompt is vague.

## STATUS
`NEEDS_USER_VISUAL_CHECK`
