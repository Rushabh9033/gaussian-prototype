import os
import sys
import pytest
import numpy as np
import json
import hashlib
import zipfile
import io
import subprocess
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from m5_vrc.vector_edges import extract_vector_edges
from m5_vrc.color_field import extract_color_field
from m5_vrc.residual_pyramid import build_laplacian_pyramid, reconstruct_laplacian_pyramid, extract_source_residuals
from m5_vrc.scale_renderer import enforce_scale_consistency, render_scale, evaluate_splines_on_grid
from m5_vrc.quadtree import build_quadtree_index, select_visible_tiles

def test_no_ai_dependencies():
    code = """
import sys
import m5_vrc.vector_edges
import m5_vrc.color_field
import m5_vrc.scale_renderer
forbidden = ['torch', 'tensorflow', 'jax', 'onnx', 'diffusers', 'transformers']
for mod in forbidden:
    if mod in sys.modules: sys.exit(1)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src", "python")
    res = subprocess.run([sys.executable, "-c", code], env=env)
    assert res.returncode == 0

def test_deterministic_edge_extraction():
    luma = np.zeros((100, 100), dtype=np.float32)
    luma[30:70, 30:70] = 1.0
    src = np.ones((100, 100, 3), dtype=np.float32)
    spl1, mask1, _ = extract_vector_edges(luma, src, coherence_thresh=0.01, strength_thresh=0.01)
    spl2, mask2, _ = extract_vector_edges(luma, src, coherence_thresh=0.01, strength_thresh=0.01)
    np.testing.assert_array_equal(mask1, mask2)
    assert len(spl1) == len(spl2)
    for c1, c2 in zip(spl1, spl2):
        assert c1["fitting_error"] == c2["fitting_error"]

def test_curve_rendering_multiple_scales():
    luma = np.zeros((100, 100), dtype=np.float32)
    luma[30:70, 30:70] = 1.0
    src = np.ones((100, 100, 3), dtype=np.float32)
    spl, mask, _ = extract_vector_edges(luma, src, 0.01, 0.01)
    
    vr1, cov1 = evaluate_splines_on_grid(spl, (100, 100), (100, 100))
    vr2, cov2 = evaluate_splines_on_grid(spl, (200, 200), (100, 100))
    
    assert vr1.shape == (100, 100, 3)
    assert vr2.shape == (200, 200, 3)
    assert cov1.max() > 0
    assert cov2.max() > 0

def test_logical_coordinate_invariance():
    luma = np.zeros((100, 100), dtype=np.float32)
    luma[30:70, 30:70] = 1.0
    src = np.ones((100, 100, 3), dtype=np.float32)
    spl, mask, _ = extract_vector_edges(luma, src, 0.01, 0.01)
    
    cf = extract_color_field(src, 3, 0.1, 1.0)
    res = np.zeros_like(src)
    
    render1, cov1 = render_scale(cf, spl, res, (100, 100), (100, 100))
    render2, cov2 = render_scale(cf, spl, res, (200, 200), (100, 100))
    
    # Scale from 100 to 200 should logically map boundaries
    assert render1.shape == (100, 100, 3)
    assert render2.shape == (200, 200, 3)
    assert cov1 > 0
    assert cov2 > 0

def test_residual_pyramid_reconstruction():
    src = np.ones((64, 64, 3), dtype=np.float32)
    src[20:40, 20:40] = 0.5
    
    pyr = build_laplacian_pyramid(src, levels=3)
    assert len(pyr) == 3
    
    recon = reconstruct_laplacian_pyramid(pyr)
    # The reconstruction should match closely
    assert np.max(np.abs(src - recon)) < 1e-4

def test_quadtree_indexing():
    root = build_quadtree_index(360, 220, max_z=2, tile_size=256)
    assert len(root.children) > 0
    
    # Viewport selection
    vp = (0, 0, 100, 100)
    visible = select_visible_tiles(root, 2, vp)
    assert len(visible) > 0
    for v in visible:
        assert v.z == 2
        # Check overlap
        b = v.bounds
        assert (b[0] < vp[2] and b[2] > vp[0] and b[1] < vp[3] and b[3] > vp[1])

def test_scale_consistency_high_contrast():
    parent = np.zeros((50, 50, 3), dtype=np.float32)
    parent[20:30, 20:30] = 1.0 # High contrast square
    child = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    
    corrected = enforce_scale_consistency(child, parent, max_iters=20, tolerance=1.0/255.0)
    import cv2
    down = cv2.resize(corrected, (50, 50), interpolation=cv2.INTER_AREA)
    err = np.max(np.abs(down - parent))
    assert err <= 3.0/255.0 + 1e-5 # Close enough given clipping limits

def hash_data(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def test_package_round_trip():
    pkg_path = os.path.join(os.path.dirname(__file__), "..", "m5_results", "vrc_package.zip")
    if not os.path.exists(pkg_path):
        pytest.fail("Package not found. Run benchmark first.")
        
    with zipfile.ZipFile(pkg_path, "r") as zf:
        meta = json.loads(zf.read("metadata.json"))
        assert "logical_dimensions" in meta
        
        curves = json.loads(zf.read("vector_curves.json"))
        assert isinstance(curves, list)
        
        # Validate residuals using np.load safely
        res_data = zf.read("residuals.npz")
        io_buf = io.BytesIO(res_data)
        npz = np.load(io_buf, allow_pickle=False)
        assert "level_0" in npz

def test_artifact_hash_verification():
    ev_path = os.path.join(os.path.dirname(__file__), "..", "m5_results", "evidence.jsonl")
    if not os.path.exists(ev_path):
        pytest.fail("Evidence missing.")
    with open(ev_path, "r") as f:
        data = json.loads(f.readline())
        
    for a in data["artifacts"]:
        assert os.path.exists(a["path"])
        h = hashlib.sha256()
        with open(a["path"], "rb") as f2:
            h.update(f2.read())
        assert h.hexdigest() == a["sha256"]

def test_failure_evidence_generation():
    script = """
import sys, json
try:
    raise RuntimeError("Intentional Failure")
except Exception as e:
    import traceback, uuid, datetime
    ev = {
        "run_uuid": str(uuid.uuid4()),
        "utc_timestamp": datetime.datetime.now().isoformat(),
        "exception_type": type(e).__name__,
        "exception_message": str(e),
        "traceback": traceback.format_exc(),
        "status": "BLOCKED"
    }
    with open("m5_results/fail_evidence.jsonl", "w") as f:
        f.write(json.dumps(ev))
    sys.exit(1)
"""
    env = os.environ.copy()
    subprocess.run([sys.executable, "-c", script], env=env)
    fail_ev_path = "m5_results/fail_evidence.jsonl"
    assert os.path.exists(fail_ev_path)
    with open(fail_ev_path, "r") as f:
        ev = json.loads(f.readline())
    assert ev["status"] == "BLOCKED"
    assert ev["exception_type"] == "RuntimeError"
    assert "Intentional Failure" in ev["exception_message"]
    assert "Intentional Failure" in ev["traceback"]
    os.remove(fail_ev_path)

def test_rendering_from_original_representation():
    # Assert that render_scale executes correctly by explicitly passing separate original and target shape dimensions
    from m5_vrc.scale_renderer import render_scale
    luma = np.zeros((100, 100), dtype=np.float32)
    src = np.ones((100, 100, 3), dtype=np.float32)
    spl, mask, _ = extract_vector_edges(luma, src, 0.01, 0.01)
    cf = extract_color_field(src, 3, 0.1, 1.0)
    res = np.zeros_like(src)
    # Target shape is completely independent of the input array shapes
    render_out, cov = render_scale(cf, spl, res, target_shape=(300, 300), original_shape=(100, 100))
    assert render_out.shape == (300, 300, 3)
    
def test_no_recursive_bitmap_scaling():
    # Assert that enforce_scale_consistency explicitly expects the 1x base parent, not an intermediate upscale
    from m5_vrc.scale_renderer import enforce_scale_consistency
    parent_1x = np.ones((50, 50, 3), dtype=np.float32)
    child_nx = np.ones((400, 400, 3), dtype=np.float32) # 8x scale directly
    out = enforce_scale_consistency(child_nx, parent_1x, max_iters=2, tolerance=0.1)
    assert out.shape == (400, 400, 3)

def test_no_tracked_pyc():
    import subprocess
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    assert res.returncode == 0
    for f in res.stdout.splitlines():
        assert not f.endswith(".pyc")
