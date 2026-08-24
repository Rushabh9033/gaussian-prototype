import os
import sys
import pytest
import numpy as np
import cv2
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
from m5b_edr.structure_tensor import extract_edge_orientation
from m5b_edr.coordinate_map import generate_output_coordinates
from m5b_edr.anisotropic_sampler import sample_anisotropic
from m5b_edr.tiled_renderer import render_tiled
from m5b_edr.metrics import compute_tiling_equivalence_error, compute_edge_orientation_error

def test_no_ai_dependencies():
    import subprocess
    code = "import sys; import m5b_edr.tiled_renderer; " \
           "forbidden = ['torch', 'tensorflow', 'jax', 'onnx', 'diffusers']; " \
           "sys.exit(any(mod in sys.modules for mod in forbidden))"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src", "python")
    res = subprocess.run([sys.executable, "-c", code], env=env)
    assert res.returncode == 0

def test_horizontal_edge_orientation():
    luma = np.zeros((20, 20), dtype=np.float32)
    luma[10:, :] = 1.0
    tensor = extract_edge_orientation(luma, sigma=0)
    theta = tensor["theta"][10, 10]
    assert np.isclose(theta, 0, atol=1e-5) or np.isclose(np.abs(theta), np.pi, atol=1e-5)

def test_vertical_edge_orientation():
    luma = np.zeros((20, 20), dtype=np.float32)
    luma[:, 10:] = 1.0
    tensor = extract_edge_orientation(luma, sigma=0)
    theta = tensor["theta"][10, 10]
    assert np.isclose(np.abs(theta), np.pi/2, atol=1e-5)

def test_diagonal_edge_orientation():
    luma = np.zeros((20, 20), dtype=np.float32)
    for i in range(20):
        for j in range(20):
            if i > j:
                luma[i, j] = 1.0
    tensor = extract_edge_orientation(luma, sigma=0)
    theta = tensor["theta"][10, 10]
    # Gradient is along diagonal -> normal is 45 deg or 225 deg. Tangent is -45 or 135 deg (pi/4)
    # The absolute angle should be pi/4 or 3*pi/4
    assert np.isclose(np.abs(theta), np.pi/4, atol=1e-1) or np.isclose(np.abs(theta), 3*np.pi/4, atol=1e-1)

def test_linear_gradient_preservation():
    src = np.zeros((10, 10, 3), dtype=np.float32)
    for x in range(10):
        src[:, x, :] = x / 9.0
    theta = np.zeros((10, 10), dtype=np.float32)
    coh = np.zeros((10, 10), dtype=np.float32)
    unc = np.zeros((10, 10), dtype=np.float32)
    cfg = {"tangent_scale": 1.0, "normal_scale": 1.0, "coherence_thresh": 0.5, "photometric_sigma": 1.0}
    # Rendering to same size should approximate the linear gradient
    out = render_tiled(src, 10, 10, theta, coh, unc, cfg, tile_size=10)
    # the center of the gradient shouldn't deviate wildly
    assert np.allclose(out[:, 4:6, :], src[:, 4:6, :], atol=0.1)

def test_constant_color_preservation():
    src = np.ones((10, 10, 3), dtype=np.float32) * 0.5
    theta = np.zeros((10, 10), dtype=np.float32)
    coh = np.zeros((10, 10), dtype=np.float32)
    unc = np.zeros((10, 10), dtype=np.float32)
    cfg = {"tangent_scale": 1.5, "normal_scale": 0.8, "coherence_thresh": 0.5, "photometric_sigma": 0.2}
    out = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=10)
    assert np.allclose(out, 0.5, atol=1e-5)

def test_deterministic_output():
    src = np.random.rand(10, 10, 3).astype(np.float32)
    theta = np.zeros((10, 10), dtype=np.float32)
    coh = np.zeros((10, 10), dtype=np.float32)
    unc = np.zeros((10, 10), dtype=np.float32)
    cfg = {"tangent_scale": 1.5, "normal_scale": 0.8, "coherence_thresh": 0.5, "photometric_sigma": 0.2}
    out1 = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=10)
    out2 = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=10)
    np.testing.assert_array_equal(out1, out2)

def test_tile_order_invariance():
    src = np.random.rand(10, 10, 3).astype(np.float32)
    theta = np.zeros((10, 10), dtype=np.float32)
    coh = np.zeros((10, 10), dtype=np.float32)
    unc = np.zeros((10, 10), dtype=np.float32)
    cfg = {"tangent_scale": 1.5, "normal_scale": 0.8, "coherence_thresh": 0.5, "photometric_sigma": 0.2}
    
    # Standard tile order (y, x) is implemented in render_tiled.
    # To test order invariance, we just render one tile out-of-order manually
    x_in, y_in = generate_output_coordinates(20, 20, 10, 10, start_x=10, start_y=10, tile_w=10, tile_h=10)
    tile_manual = sample_anisotropic(src, x_in, y_in, theta, coh, unc, cfg, k_radius=2)
    
    out_full = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=10)
    tile_from_full = out_full[10:20, 10:20]
    
    np.testing.assert_array_equal(tile_manual, tile_from_full)

def test_tiled_vs_untiled_equality():
    src = np.random.rand(10, 10, 3).astype(np.float32)
    theta = np.zeros((10, 10), dtype=np.float32)
    coh = np.zeros((10, 10), dtype=np.float32)
    unc = np.zeros((10, 10), dtype=np.float32)
    cfg = {"tangent_scale": 1.5, "normal_scale": 0.8, "coherence_thresh": 0.5, "photometric_sigma": 0.2}
    
    out_full = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=20)
    out_tiled = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=10)
    
    max_err, mean_err = compute_tiling_equivalence_error(out_full, out_tiled)
    assert max_err < 1e-5

def test_signed_difference_correctness():
    # uint8 subtraction wrap: 255 - 0 = 255 but using normal uint8 math 0 - 255 = 1 (wraps).
    # The new metric uses float32.
    img1 = np.array([0, 255], dtype=np.uint8)
    img2 = np.array([255, 0], dtype=np.uint8)
    max_err, mean_err = compute_tiling_equivalence_error(img1, img2)
    assert max_err == 255.0
    assert mean_err == 255.0

def test_cross_edge_color_bleeding():
    src = np.zeros((10, 10, 3), dtype=np.float32)
    src[:, 5:, :] = 1.0 # Sharp vertical edge
    theta = np.zeros((10, 10), dtype=np.float32)
    theta[:, :] = np.pi/2 # tangent is vertical
    coh = np.ones((10, 10), dtype=np.float32)
    unc = np.zeros((10, 10), dtype=np.float32)
    
    # Photometric sigma large (allows bleeding)
    cfg_bleed = {"tangent_scale": 2.0, "normal_scale": 2.0, "coherence_thresh": 0.0, "photometric_sigma": 10.0}
    out_bleed = render_tiled(src, 10, 10, theta, coh, unc, cfg_bleed, tile_size=10)
    
    # Photometric sigma small (prevents bleeding)
    cfg_no_bleed = {"tangent_scale": 2.0, "normal_scale": 2.0, "coherence_thresh": 0.0, "photometric_sigma": 0.01}
    out_no_bleed = render_tiled(src, 10, 10, theta, coh, unc, cfg_no_bleed, tile_size=10)
    
    # Bleed case should have more intermediate values at the boundary column (x=4 and x=5)
    bleed_diff = np.abs(out_bleed[:, 4, :] - 0.5).mean()
    no_bleed_diff = np.abs(out_no_bleed[:, 4, :] - 0.5).mean()
    assert no_bleed_diff > bleed_diff # no_bleed is closer to 0 or 1, not 0.5

def test_metrics_correctness():
    # Orientation error
    t_gt = np.zeros((10, 10), dtype=np.float32)
    t_pr = np.ones((10, 10), dtype=np.float32) * (np.pi / 2)
    s_gt = np.ones((10, 10), dtype=np.float32)
    err_rad, err_deg = compute_edge_orientation_error(t_gt, t_pr, s_gt)
    assert np.isclose(err_rad, np.pi / 2)
    assert np.isclose(err_deg, 90.0)

def test_benchmark_owned_failure_evidence():
    import subprocess
    script = """
import sys, os
sys.path.insert(0, os.getcwd())
import run_m5b_benchmark
def fail_process(*args, **kwargs):
    raise ValueError("Real benchmark failure")
run_m5b_benchmark.process_edr = fail_process
run_m5b_benchmark.get_git_dirty = lambda: False
try:
    run_m5b_benchmark.main()
except SystemExit:
    pass
"""
    env = os.environ.copy()
    subprocess.run([sys.executable, "-c", script], env=env)
    
    assert os.path.exists("m5b_results/evidence_fail.json")
    with open("m5b_results/evidence_fail.json", "r") as f:
        ev = json.load(f)
        
    assert ev["exception_type"] == "ValueError"
    assert "Real benchmark failure" in ev["exception_message"]
    assert "Real benchmark failure" in ev["traceback"]
    os.remove("m5b_results/evidence_fail.json")
