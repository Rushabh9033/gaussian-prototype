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
from m5b_edr.metrics import compute_tile_seam_error, compute_edge_orientation_error

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

def test_output_coordinate_mapping():
    x, y = generate_output_coordinates(10, 10, 5, 5, start_x=0, start_y=0, tile_w=10, tile_h=10)
    assert np.isclose(x[0, 0], -0.25)
    assert np.isclose(x[9, 9], 4.25)

def test_tile_overlap_seam_free():
    src = np.random.rand(10, 10, 3).astype(np.float32)
    theta = np.zeros((10, 10), dtype=np.float32)
    coh = np.zeros((10, 10), dtype=np.float32)
    unc = np.zeros((10, 10), dtype=np.float32)
    cfg = {"tangent_scale": 1.5, "normal_scale": 0.8, "coherence_thresh": 0.5, "photometric_sigma": 0.2}
    
    out_full = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=20)
    out_tiled = render_tiled(src, 20, 20, theta, coh, unc, cfg, tile_size=10)
    np.testing.assert_allclose(out_full, out_tiled, atol=1e-5)

def test_corner_isotropic_fallback():
    x, y = np.array([[5.0]], dtype=np.float32), np.array([[5.0]], dtype=np.float32)
    src = np.zeros((10, 10, 3), dtype=np.float32)
    src[5, 5] = 1.0
    
    cfg = {"tangent_scale": 10.0, "normal_scale": 0.1, "coherence_thresh": 0.0, "photometric_sigma": 1.0}
    theta = np.zeros((10, 10), dtype=np.float32)
    coh = np.ones((10, 10), dtype=np.float32)
    unc = np.ones((10, 10), dtype=np.float32)
    
    out_corner = sample_anisotropic(src, x, y, theta, coh, unc, cfg, k_radius=2)
    
    coh_zero = np.zeros((10, 10), dtype=np.float32)
    unc_zero = np.zeros((10, 10), dtype=np.float32)
    out_flat = sample_anisotropic(src, x, y, theta, coh_zero, unc_zero, cfg, k_radius=2)
    
    np.testing.assert_allclose(out_corner, out_flat, atol=1e-5)

def test_metrics_correctness():
    # Tile seam error
    img = np.zeros((20, 20, 3), dtype=np.float32)
    img[:, 10:] = 1.0
    err_max, err_mean = compute_tile_seam_error(img, tile_size=10)
    assert err_max == 1.0
    
    # Orientation error
    t_gt = np.zeros((10, 10), dtype=np.float32)
    t_pr = np.ones((10, 10), dtype=np.float32) * (np.pi / 2)
    mask = np.ones((10, 10), dtype=np.uint8)
    err = compute_edge_orientation_error(t_gt, t_pr, mask)
    assert np.isclose(err, np.pi / 2)
    
def test_real_failure_evidence_with_traceback():
    import subprocess
    script = """
import sys, os
sys.path.insert(0, os.getcwd())
import run_m5b_benchmark
def fail_process(*args, **kwargs):
    raise ValueError("Real benchmark failure")
run_m5b_benchmark.process_edr = fail_process
try:
    run_m5b_benchmark.main()
except Exception as e:
    import json, traceback, uuid, datetime
    ev = {
        "exception_type": type(e).__name__,
        "exception_message": str(e),
        "traceback": traceback.format_exc()
    }
    with open("m5b_results/test_fail.json", "w") as f:
        json.dump(ev, f)
"""
    env = os.environ.copy()
    subprocess.run([sys.executable, "-c", script], env=env)
    
    assert os.path.exists("m5b_results/test_fail.json")
    with open("m5b_results/test_fail.json", "r") as f:
        ev = json.load(f)
        
    assert ev["exception_type"] == "ValueError"
    assert "Real benchmark failure" in ev["exception_message"]
    assert "Real benchmark failure" in ev["traceback"]
    os.remove("m5b_results/test_fail.json")
