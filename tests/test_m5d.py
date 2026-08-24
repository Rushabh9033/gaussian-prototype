import pytest
import numpy as np
import cv2
from src.python.m5d_scpb.psf_estimator import estimate_psf
from src.python.m5d_scpb.forward_model import forward_model
from src.python.m5d_scpb.backprojection import compute_objective, backproject_residual, run_scpb_optimization
from src.python.m5d_scpb.regularization import apply_regularization
from src.python.m5d_scpb.tiling import reconstruct_tiled
from run_m5d_benchmark import srgb_to_lin, lin_to_srgb

def test_srgb_linear_roundtrip():
    test_val = (np.random.rand(10, 10, 3) * 255).astype(np.float32)
    lin = srgb_to_lin(test_val)
    srgb = lin_to_srgb(lin) * 255.0
    np.testing.assert_allclose(test_val, srgb, atol=1e-1)

def test_psf_properties():
    img = np.ones((50, 50, 3), dtype=np.float32) * 0.5
    img[:, 25:] = 0.9
    psf_res = estimate_psf(img)
    if psf_res['confidence'] > 0:
        kernel = psf_res['kernel']
        assert kernel is not None
        assert np.isclose(np.sum(kernel), 1.0)
        np.testing.assert_allclose(kernel, kernel.T)
        assert np.all(kernel >= 0)

def test_psf_rejection():
    img = np.random.rand(50, 50, 3).astype(np.float32)
    psf_res = estimate_psf(img)
    assert psf_res['confidence'] == 0.0

def test_forward_model():
    hr = np.random.rand(40, 40).astype(np.float32)
    source_shape = (20, 20)
    out = forward_model(hr, source_shape, 2, 0.5)
    assert out.shape == source_shape

def test_backprojection():
    res = np.random.rand(20, 20).astype(np.float32)
    out = backproject_residual(res, (40, 40), 1.0)
    assert out.shape == (40, 40)

def test_tiling_equivalence():
    lr = np.ones((30, 30, 3), dtype=np.float32) * 0.5
    hr_init = cv2.resize(lr, (60, 60), interpolation=cv2.INTER_LANCZOS4)
    cfg = {"psf_multiplier": 1.0, "max_iters": 2, "initial_step": 0.5, "reg_weight": 0.0, "overshoot_weight": 0.0}
    out_untiled, _ = run_scpb_optimization(lr, hr_init, 2, 0.5, cfg, 2, 0.5)
    out_tiled = reconstruct_tiled(lr, 2, 0.5, cfg, 2, 0.5, tile_size=16, padding=16)
    diff = np.abs(out_untiled - out_tiled)
    assert np.max(diff) < 1e-3
    
def test_objective_non_increase():
    lr = np.ones((20, 20, 3), dtype=np.float32) * 0.5
    hr = cv2.resize(lr, (40, 40), interpolation=cv2.INTER_LANCZOS4)
    cfg = {"psf_multiplier": 1.0, "max_iters": 5, "initial_step": 0.5, "reg_weight": 0.01, "overshoot_weight": 10.0}
    out, history = run_scpb_optimization(lr, hr, 2, 0.5, cfg, 5, 0.5)
    
    if len(history) > 0:
        objs = [h['objective'] for h in history if h['accepted']]
        for i in range(1, len(objs)):
            assert objs[i] <= objs[i-1] + 1e-6
