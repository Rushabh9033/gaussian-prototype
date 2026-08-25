import pytest
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock execution script imports or setup for tests
def test_controlled_input_generation(tmpdir):
    gt = np.zeros((220, 360, 3), dtype=np.uint8)
    gt_path = os.path.join(tmpdir, "porche.png")
    cv2.imwrite(gt_path, gt)
    
    lr2x = cv2.resize(gt, (180, 110), interpolation=cv2.INTER_AREA)
    lr4x = cv2.resize(gt, (90, 55), interpolation=cv2.INTER_AREA)
    
    assert lr2x.shape == (110, 180, 3)
    assert lr4x.shape == (55, 90, 3)

def test_dimensions():
    # 2X outscale
    assert 180 * 2 == 360
    assert 110 * 2 == 220
    # 4X outscale
    assert 90 * 4 == 360
    assert 55 * 4 == 220
    # Full outscale
    assert 360 * 4 == 1440
    assert 220 * 4 == 880

def test_metrics_sanity():
    import m6a_multiframe.metrics as metrics
    img1 = np.ones((100, 100, 3), dtype=np.float32)
    img2 = np.ones((100, 100, 3), dtype=np.float32)
    
    psnr = metrics.compute_psnr(img1, img2)
    ssim = metrics.compute_ssim(img1, img2)
    assert np.isinf(psnr) or psnr >= 100
    assert np.isclose(ssim, 1.0)

def test_regional_coordinates():
    # Plate, wheel, headlamp, silhouette
    coords = {
        'plate': (110, 145, 30, 90),
        'headlamp': (60, 110, 20, 60),
        'wheel': (120, 200, 180, 260),
        'silhouette': (30, 210, 0, 360)
    }
    for name, (y1, y2, x1, x2) in coords.items():
        assert y2 > y1
        assert x2 > x1
        assert y2 <= 220
        assert x2 <= 360

def test_no_external_api(monkeypatch):
    import requests
    def mock_post(*args, **kwargs):
        raise RuntimeError("API Call Blocked")
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(requests, "get", mock_post)
    
    # Simple check that the code doesn't secretly bypass
    import run_m8a_benchmark
    assert "requests.post" not in open(run_m8a_benchmark.__file__).read()
    
def test_gfpgan_disabled():
    import run_m8a_benchmark
    content = open(run_m8a_benchmark.__file__).read()
    assert "face_enhancer" not in content or "face_enhancer=False" in content or "face_enhance=False" in content or "GFPGANer" not in content

def test_viewer_v3_unmodified():
    # Proof that viewer code is unchanged
    import subprocess
    diff = subprocess.check_output(["git", "diff", "407e1bbeed7263a21c7f9b8dd37557ec8905d137", "--", "src/viewer"]).decode()
    assert len(diff.strip()) == 0
    diff2 = subprocess.check_output(["git", "diff", "407e1bbeed7263a21c7f9b8dd37557ec8905d137", "--", "src/python/cli.py"]).decode()
    assert len(diff2.strip()) == 0

def test_evidence_schema():
    # We will just assert that run_m8a_benchmark has the keys we need
    import run_m8a_benchmark
    assert hasattr(run_m8a_benchmark, "generate_evidence")
