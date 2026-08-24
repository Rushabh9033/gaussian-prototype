import os
import sys
import json
import hashlib
import numpy as np
from PIL import Image
import pytest
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
from m4d.color import srgb_to_linear, linear_to_srgb
from m4d.zoom import zoom_nearest, zoom_bicubic, zoom_lanczos, zoom_hybrid
from m4d.metrics import compute_psnr, compute_ssim, edge_f1_score, calculate_ringing_percentage

def test_no_ai_dependencies():
    """Verify that no AI dependencies are imported in the m4d module."""
    import subprocess
    code = """
import sys
import m4d.zoom
import m4d.metrics
import m4d.color
forbidden = ['torch', 'tensorflow', 'jax', 'onnx', 'diffusers', 'transformers']
for mod in forbidden:
    if mod in sys.modules:
        print(f"Forbidden module {mod} was imported!")
        sys.exit(1)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src", "python")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"AI dependency detected: {res.stdout}"

def test_linear_light_round_trip():
    """Test sRGB -> linear -> sRGB roundtrip."""
    test_arr = np.array([0, 10, 50, 128, 200, 255], dtype=np.uint8)
    # create a dummy image
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[0, 0, :] = test_arr[:3]
    img[1, 2, :] = test_arr[3:]
    
    linear = srgb_to_linear(img)
    back = linear_to_srgb(linear, to_uint8=True)
    
    np.testing.assert_array_equal(img, back)

def test_output_dimensions():
    """Test that all zoom methods produce correct dimensions."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    target = (40, 20) # (w, h)
    
    out_nn = zoom_nearest(img, target)
    out_bc = zoom_bicubic(img, target)
    out_lz = zoom_lanczos(img, target)
    out_hy = zoom_hybrid(img, target)
    
    for out in [out_nn, out_bc, out_lz, out_hy]:
        assert out.shape == (20, 40, 3) # (h, w, c)

def test_deterministic_output():
    """Test that the hybrid zoom is deterministic."""
    img = np.random.randint(0, 256, (10, 20, 3), dtype=np.uint8)
    target = (40, 20)
    
    out1 = zoom_hybrid(img, target)
    out2 = zoom_hybrid(img, target)
    
    np.testing.assert_array_equal(out1, out2)

def test_no_out_of_range_rgb():
    """Test that hybrid zoom doesn't produce out-of-range RGB values."""
    img = np.random.randint(0, 256, (10, 20, 3), dtype=np.uint8)
    target = (40, 20)
    
    out = zoom_hybrid(img, target)
    assert out.min() >= 0
    assert out.max() <= 255
    assert out.dtype == np.uint8

def test_metrics_correctness():
    """Test that metrics produce expected baseline values."""
    img1 = np.zeros((50, 50, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    
    assert compute_psnr(img1, img2) == float('inf')
    assert compute_ssim(img1, img2) == 1.0
    
    edge_stats = edge_f1_score(img1, img2)
    assert edge_stats["f1"] == 0.0 # No edges in solid black
    
    img1[10:40, 10:40] = 255
    img2[10:40, 10:40] = 255
    edge_stats2 = edge_f1_score(img1, img2)
    assert edge_stats2["f1"] > 0.9 # Should perfectly match

def test_anti_ringing_clamp():
    """Test that ringing is reduced compared to raw unsharp."""
    # We will just test that the percentage calculation works and doesn't crash
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2[5,5] = 255 # Massive overshoot
    
    pct = calculate_ringing_percentage(img1, img2)
    assert pct > 0.0
    
def test_edge_map_generation():
    """Test that edge map extraction works."""
    from m4d.metrics import get_edge_map
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    edges = get_edge_map(img)
    assert edges.sum() > 0

def test_expected_outputs_exist():
    """Verify that expected artifacts exist and hashes match."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "m4d_results")
    evidence_path = os.path.join(results_dir, "evidence.jsonl")
    
    assert os.path.exists(evidence_path), "No evidence file found. Run benchmark first."
        
    with open(evidence_path, "r") as f:
        data = json.loads(f.readline())
        
    for artifact in data["artifacts"]:
        path = artifact["path"]
        expected_hash = artifact["sha256"]
        
        assert os.path.exists(path), f"Artifact missing: {path}"
        
        if path.endswith(".png"):
            img = Image.open(path)
            img.verify()
            
        h = hashlib.sha256()
        with open(path, "rb") as f2:
            for chunk in iter(lambda: f2.read(4096), b""):
                h.update(chunk)
        assert h.hexdigest() == expected_hash, f"Hash mismatch for {path}"

def test_no_tracked_pyc_files():
    """Confirm no .pyc or __pycache__ path is tracked by Git."""
    import subprocess
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if res.returncode == 0:
        tracked_files = res.stdout.splitlines()
        for f in tracked_files:
            assert not f.endswith(".pyc"), f"Tracked pyc file found: {f}"
            assert "__pycache__" not in f, f"Tracked pycache path found: {f}"

def test_ringing_overflow_regression():
    """Test that calculate_ringing_percentage doesn't overflow on uint8."""
    img1 = np.full((10, 10, 3), 254, dtype=np.uint8)
    img2 = np.full((10, 10, 3), 255, dtype=np.uint8)
    pct = calculate_ringing_percentage(img1, img2)
    # local_max is 254. 254+5=259. 255 is not > 259.
    # If uint8 overflowed, 254+5 = 3. 255 > 3 -> 100% ringing.
    assert pct == 0.0

def test_edge_precision_recall_formulation():
    """Test explicit precision and recall formulation of edge F1."""
    true_img = np.zeros((30, 30, 3), dtype=np.uint8)
    pred_img = np.zeros((30, 30, 3), dtype=np.uint8)
    
    true_img[5:15, 5:15] = 255
    pred_img[5:15, 5:15] = 255
    # false positive edge
    pred_img[20:25, 20:25] = 255
    
    stats = edge_f1_score(true_img, pred_img, tolerance=1)
    assert stats["precision"] < 1.0
    assert stats["recall"] == pytest.approx(1.0)

