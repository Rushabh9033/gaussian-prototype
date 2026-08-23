# Milestone 4B: Bounded Alternative Deep-Zoom Architecture Bake-Off

## Status: BLOCKED

### Goal
To find whether a fundamentally different architecture can generate five coherent, sharp, synthetic deep-zoom levels from one normal photograph (the Porsche image, `128x128` region centered at `245, 150`, zooming up to 256x).

### Hardware Limitations
The host system is equipped with an **NVIDIA GeForce RTX 4060 Laptop GPU** with **8.59 GB of VRAM**.

### Candidate A — Official Chain-of-Zoom
- **Status:** **BLOCKED**
- **Reason:** Hardware limitations. The official `Chain-of-Zoom` implementation requires 24GB VRAM even in its `--efficient_memory` mode to run SD3 Medium + Qwen2.5-VL-3B-Instruct + RAM simultaneously. The available 8.59 GB VRAM is insufficient.

### Candidate B — Joint Multi-Scale Generation (Generative Powers of Ten)
- **Status:** **BLOCKED_NO_REPRODUCIBLE_IMPLEMENTATION**
- **Reason:** The official authors (Google/UW) did not release source code for "Generative Powers of Ten" (arXiv:2312.02149). The only available implementation (`atfortes/generative-powers-of-ten`) is unofficial and uses DeepFloyd IF, which requires a gated HuggingFace research license (which we cannot accept automatically) and vastly exceeds our 8.59 GB VRAM limit.

### Candidate C — Direct-from-Root Structure-Conditioned Generation
- **Status:** **BLOCKED**
- **Reason:** Hardware limitations. A custom pipeline requiring a local Vision-Language Model (VLM) for material descriptions, a depth/edge estimator, and a structural-conditioned image-to-image model (e.g., ControlNet) cannot fit into the 8.59 GB VRAM without severe, untested aggressive offloading. Furthermore, the necessary multi-gigabyte checkpoints for these components are not locally available, and attempting to download them would violate bounded execution constraints.

### Conclusion
No candidate could be safely executed under the current hardware constraints and licensing restrictions. No mock images or fake checkpoints were generated. Milestone 4B is blocked pending hardware upgrades or checkpoint availability.
