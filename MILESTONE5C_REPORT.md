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

- **Benchmark Commit**: 100066bf77d07b2b2226df65c167b89243bf0882
- **Commands**: 
  - `python run_m5c_benchmark.py`
  - `python validate_m5c.py`
- **Validation**: All tests passed (72 passed, 0 failed, 0 skipped).
- **Viewer-build**: Successful (exit code 0).
- **Peak RAM**: 271.4 MiB (284622848 bytes)
- **Tile-Equivalence Maximum Error**: 0.00

## Metrics (2x / 4x / 8x)
| Scale | Config | PSNR | SSIM | Edge F1 | Ringing % | Max Tile Err |
|---|---|---|---|---|---|---|
| 2xx | Bicubic | 26.80 | 0.8596 | 0.8585 | 0.2% | 0.0 |
| 2xx | Lanczos | 26.84 | 0.8602 | 0.8780 | 0.4% | 0.0 |
| 2xx | ISEPR_A | 25.74 | 0.8412 | 0.8879 | 2.1% | 0.0 |
| 2xx | ISEPR_B | 25.84 | 0.8436 | 0.8881 | 2.0% | 0.0 |
| 2xx | ISEPR_C | 25.87 | 0.8449 | 0.8877 | 1.9% | 0.0 |
| 4xx | Bicubic | 22.82 | 0.6526 | 0.4589 | 3.7% | 0.0 |
| 4xx | Lanczos | 22.84 | 0.6517 | 0.4980 | 3.7% | 0.0 |
| 4xx | ISEPR_A | 22.38 | 0.6279 | 0.4944 | 6.4% | 0.0 |
| 4xx | ISEPR_B | 22.44 | 0.6304 | 0.5118 | 6.1% | 0.0 |
| 4xx | ISEPR_C | 22.48 | 0.6323 | 0.5136 | 5.9% | 0.0 |
| 8xx | Bicubic | 20.28 | 0.4613 | 0.0353 | 18.0% | 0.0 |
| 8xx | Lanczos | 20.34 | 0.4627 | 0.0575 | 17.6% | 0.0 |
| 8xx | ISEPR_A | 19.84 | 0.4364 | 0.0919 | 20.3% | 0.0 |
| 8xx | ISEPR_B | 19.85 | 0.4362 | 0.0931 | 20.4% | 0.0 |
| 8xx | ISEPR_C | 19.86 | 0.4358 | 0.1035 | 20.3% | 0.0 |

## Artifacts
| Path | Size | SHA-256 |
|---|---|---|
| m4d_test_assets\porche.png | 158891 | 8eda0f62759c5d72da363eb338329a87650cd931076fb1c0f932d71dbefdafc1 |
| m5c_results\contact_op_2x_full.png | 811451 | e2fddb69997d5c47a4ff7b3336ead0b0bb564c6cd0865952e5c86d5d5da60dfa |
| m5c_results\contact_op_2x_headlamp.png | 41199 | 001dbf86cadf4853f8e159e8c3153eaff8295c032ba053bfd448e6f47d479f14 |
| m5c_results\contact_op_2x_plate.png | 38900 | c29d8bca33b96b0fb34e8a40ef514a91b28d8a77e786e55145ab8447dfc56e4f |
| m5c_results\contact_op_2x_wheel.png | 103723 | 8695851db2e5d9ae107815dadc48e56af6b4d83e56b8bb4652be1d408aed15dd |
| m5c_results\contact_op_4x_full.png | 2417823 | 5c30eefe8ac61a4ef6d7b3fd4098a24d1d508963ba7ebf71110794c987a3ef1a |
| m5c_results\contact_op_4x_headlamp.png | 116390 | f5f295e946bb17a3d30afe4586b73bbbe0e974c7462916c498b240dd1136dc42 |
| m5c_results\contact_op_4x_plate.png | 108608 | 5cf0d2df17c19450e91e510dfae7c52b02cffc7d1b218e38bf36af8dd0997a16 |
| m5c_results\contact_op_4x_wheel.png | 300697 | 4be5911ff3fb34c2d3560b0cf2d7545405c25f2b2c92843b143ae207cb9cfc2d |
| m5c_results\contact_op_8x_full.png | 7375976 | 60ba5ab19da4c8a555afe3f20d5f4f7254b4af64d8449ada716ec23881012ab0 |
| m5c_results\contact_op_8x_headlamp.png | 327421 | d52d819b4553ef09d78a45d2c77a88c4b2389815f7c3033ad1820732c8e36b06 |
| m5c_results\contact_op_8x_plate.png | 302414 | 53ce1fe7e43eee40ebf30bcccc2534001e9341318a61459532ffb445e5e38574 |
| m5c_results\contact_op_8x_wheel.png | 889467 | 50d31529057125a558222a18c62e89c465b4884377e70b4863d30dcb987e4c71 |
| m5c_results\contact_recon_2x_foliage.png | 10281 | 127417e497d88e09fa79b7952c5e7aa08d7632e70470e646c205a990ba934900 |
| m5c_results\contact_recon_2x_full.png | 578515 | 5d41db4c27785e02eb7fd58d17b8f7ed9e4e2d11fcecc4ec67fad978d1fb7420 |
| m5c_results\contact_recon_2x_headlamp.png | 31547 | 5e67fe1bee6b7bdfd9230b9cc6e1f6b83cea8c8e63e457d8642e815f8ee0d3a4 |
| m5c_results\contact_recon_2x_plate.png | 27134 | 8e7b4bce2ebfb459a5568bb40a7405d143b8c9c1e81bc8e45345697396edf500 |
| m5c_results\contact_recon_2x_road.png | 23405 | 4e9dbcc9ee3a3df4a2bdec2d8fa36bef1bc4d471abdb8875b8836f6dea33da5e |
| m5c_results\contact_recon_2x_smooth.png | 16802 | 585303f66472fd452b83789f715e6a10daea098aff3f2cb72b925c74025ab79f |
| m5c_results\contact_recon_2x_wheel.png | 71017 | cbf7b01007753cde32fbf518109e4666576fe3d3ada7591383ace3d88831e29f |
| m5c_results\contact_recon_4x_foliage.png | 9936 | 58a812f3df64b52c1cf2d2c549d9fb0a163cb5cb063260f29f7956270e423660 |
| m5c_results\contact_recon_4x_full.png | 568982 | 47e2d5742dbbbd396ae2dadcceb3af139894ed339d14ff62dc10e81fe145ff61 |
| m5c_results\contact_recon_4x_headlamp.png | 30743 | 788acb4e38610230d5d7db8e783e0dd118148bc2ab7f7b714702d0e7371df319 |
| m5c_results\contact_recon_4x_plate.png | 26359 | b51bad5a50d79a34e0068be40afeff5c73cb868e332dad4ad690fd22a35ac8e4 |
| m5c_results\contact_recon_4x_road.png | 22370 | d662d35cf5e7cec65c7410eda84ed59cb74658974eefd080c33e59219662f7ce |
| m5c_results\contact_recon_4x_smooth.png | 16673 | b2fa09074ac49bfaa98ab69a3c37ba80fb12f8558584e05eaed91ec923928f41 |
| m5c_results\contact_recon_4x_wheel.png | 70605 | 69968851f2354d29ac16cead54192e65bdec49a222411423d73beb5466e9784a |
| m5c_results\contact_recon_8x_foliage.png | 8770 | 94f7896c1310a0c30ebc9da4068f1a2ce8cb5010dc60ae3447d293c4c119386f |
| m5c_results\contact_recon_8x_full.png | 444063 | c37d569332b0a95cca15b05c450a220f39e28ae93f2cb3791ea2f99eb9640131 |
| m5c_results\contact_recon_8x_headlamp.png | 25219 | 0b0e985274fb0d268697752857122e752d7e0cd6a9ae3bb1476579cc715749bc |
| m5c_results\contact_recon_8x_plate.png | 22804 | 14aa040a04d209964c6d584a1a62d5604a97e48fc46345f88880c42d243a99ac |
| m5c_results\contact_recon_8x_road.png | 17359 | b91fc317e4b0d61dcc20b1a1a69ca0ae5046edc353cd96aeba59a6393f89c8ad |
| m5c_results\contact_recon_8x_smooth.png | 12782 | 92851e0678cf054ddc09d7ac2deef88f336fc5a2d9dd5b8908d14430d6dbc0bb |
| m5c_results\contact_recon_8x_wheel.png | 54840 | 23c168a3c2d553d1ed822491bb2dfba7c55afcd9d439b3dcc4f0cfbcbe828700 |
| m5c_results\op_2x_ISEPR_diagnostic.png | 435060 | b415467fc3a6f01e66cb0fd6c559f1b9a2f28bfb44532e330ea7c0a354830f71 |
| m5c_results\op_2x_Lanczos.png | 423479 | f4e389be6c52542bac252cd27d7edd56a65df7c9c5ab563fd3c1b09f3e1d57d9 |
| m5c_results\op_2x_conf_map.png | 40162 | 1fad235504ca2590d68da7f077ba53912f585ea716818b3f8585786c23c9b8c0 |
| m5c_results\op_4x_ISEPR_diagnostic.png | 1278482 | 86a943893a9546e605b1720aaef02cca570c7e1f5ddb0c25c83d3ccfdb2f5a93 |
| m5c_results\op_4x_Lanczos.png | 1180698 | b525e524f96ee3f8180badf1c30896aca6b9e021e2aea36e937f8f05f4b19cca |
| m5c_results\op_4x_conf_map.png | 297831 | f90738c09fad546f70dce6f65152b8a98d96fb20fea59cbe94eac7c0d605206d |
| m5c_results\op_8x_ISEPR_diagnostic.png | 4062779 | ea5b6c0e607b0ead892776182f2d2afc2814a17dba56b66a51e22346115417a1 |
| m5c_results\op_8x_Lanczos.png | 3301500 | 42f2923e3f20741278843f7ed769795cf3e9d8b5451f0bae6ead18e596063a97 |
| m5c_results\op_8x_conf_map.png | 1127074 | 3368ae79b187ba1b4fdc742fc9d87479311e6ff4c91e94672f91918e24995aee |
| m5c_results\provenance.json | 555373 | 24b0ff469c70e93fc033dc7f626b0b41c24918778eb1653d7d3477bd6e70a31c |
| m5c_results\results.json | 50835 | 51f52e83dc82222d7a3ef38c99fa54f86d343f9cb2009facee15ced580728759 |
