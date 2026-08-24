import numpy as np
import cv2
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath("src/python"))
from m5d_scpb import estimate_psf, run_scpb_optimization
from m5d_scpb.tiling import reconstruct_tiled
from m5d_scpb.forward_model import forward_model
from m5d_scpb.backprojection import compute_objective
from run_m5d_benchmark import srgb_to_lin, lin_to_srgb

def test_srgb_linear_roundtrip():
    srgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    lin = srgb_to_lin(srgb)
    out = (lin_to_srgb(lin) * 255).astype(np.uint8)
    # allow small quantization differences
    assert np.max(np.abs(srgb.astype(np.float32) - out.astype(np.float32))) <= 1

def test_deterministic_forward_model():
    hr = np.random.rand(100, 100, 3).astype(np.float32)
    lr1 = forward_model(hr, (25, 25), 4, 1.0)
    lr2 = forward_model(hr, (25, 25), 4, 1.0)
    assert np.array_equal(lr1, lr2)

def generate_structured_image(size=(60, 60, 3)):
    img = np.zeros(size, dtype=np.float32)
    for y in range(size[0]):
        for x in range(size[1]):
            img[y, x] = 0.5 + 0.3 * np.sin(x/5.0) * np.cos(y/5.0) + 0.1 * np.sin((x+y)/2.0)
    # add a sharp edge
    img[:, 20:30] = 0.9
    img[10:20, :] = 0.1
    return img

def test_tiling_equivalence():
    lr = generate_structured_image((30, 30, 3))
    hr_init = cv2.resize(lr, (60, 60), interpolation=cv2.INTER_LANCZOS4)
    cfg = {"psf_multiplier": 1.0, "max_iters": 2, "initial_step": 0.3, "reg_weight": 0.05, "overshoot_weight": 10.0}
    out_untiled, _ = run_scpb_optimization(lr, hr_init, 2, 0.5, cfg, 2, 0.3)
    out_tiled, tile_count, halo, pmem = reconstruct_tiled(lr, 2, 0.5, cfg, 2, 0.3, tile_size=16)
    
    diff = np.abs(out_untiled - out_tiled)
    assert np.max(diff) < 1e-3
    assert out_untiled.shape == (60, 60, 3)

def test_tiling_two_different_sizes():
    lr = generate_structured_image((45, 27, 3))
    cfg = {"psf_multiplier": 1.0, "max_iters": 2, "initial_step": 0.3, "reg_weight": 0.05, "overshoot_weight": 10.0}
    out_tiled1, tc1, _, _ = reconstruct_tiled(lr, 8, 0.5, cfg, 2, 0.3, tile_size=64)
    out_tiled2, tc2, _, _ = reconstruct_tiled(lr, 8, 0.5, cfg, 2, 0.3, tile_size=128)
    
    diff = np.abs(out_tiled1 - out_tiled2)
    assert np.max(diff) < 1e-3
    assert tc1 > tc2

def test_objective_non_increase():
    lr = generate_structured_image((30, 30, 3))
    hr_init = cv2.resize(lr, (60, 60), interpolation=cv2.INTER_LANCZOS4)
    cfg = {"psf_multiplier": 1.0, "max_iters": 5, "initial_step": 0.5, "reg_weight": 0.01, "overshoot_weight": 10.0}
    _, history = run_scpb_optimization(lr, hr_init, 2, 0.5, cfg, 5, 0.5)
    
    objs = [h['objective'] for h in history if h['accepted']]
    for i in range(1, len(objs)):
        assert objs[i] <= objs[i-1] + 1e-5

def test_deterministic_backtracking():
    lr = generate_structured_image((30, 30, 3))
    hr_init = cv2.resize(lr, (60, 60), interpolation=cv2.INTER_LANCZOS4)
    cfg = {"psf_multiplier": 1.0, "max_iters": 5, "initial_step": 0.5, "reg_weight": 0.01, "overshoot_weight": 10.0}
    _, history1 = run_scpb_optimization(lr, hr_init, 2, 0.5, cfg, 5, 0.5)
    _, history2 = run_scpb_optimization(lr, hr_init, 2, 0.5, cfg, 5, 0.5)
    
    for h1, h2 in zip(history1, history2):
        assert h1 == h2

def test_clipping_bounds():
    lr = np.random.rand(20, 20, 3).astype(np.float32) * 2.0 - 0.5
    hr_init = cv2.resize(lr, (40, 40), interpolation=cv2.INTER_LANCZOS4)
    cfg = {"psf_multiplier": 1.0, "max_iters": 2, "initial_step": 0.5, "reg_weight": 0.01, "overshoot_weight": 10.0}
    out, _ = run_scpb_optimization(lr, hr_init, 2, 0.5, cfg, 2, 0.5)
    assert np.min(out) >= 0.0
    assert np.max(out) <= 1.0

def test_synthetic_gaussian_edge_recovery():
    # Construct a synthetic edge with a known blur
    size = 100
    img = np.zeros((size, size), dtype=np.float32)
    img[:, size//2:] = 1.0
    blur_img = cv2.GaussianBlur(img, (15, 15), 2.0)
    
    psf = estimate_psf(blur_img, min_sigma=0.5, max_sigma=3.0)
    assert psf['confidence'] > 0
    assert abs(psf['sigma'] - 2.0) < 0.5
    assert psf['accepted_count'] > 0

def test_texture_rejection():
    # Noise/texture should be rejected
    img = np.random.rand(100, 100).astype(np.float32)
    psf = estimate_psf(img)
    assert psf['confidence'] == 0.0
    assert psf['reason'] == 'Insufficient accepted edges'
