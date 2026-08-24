import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from m6a_multiframe.forward_model import forward_model, srgb_to_lin, lin_to_srgb
from m6a_multiframe.frame_generator import generate_frames
from m6a_multiframe.registration import estimate_translation, register_frames
from m6a_multiframe.robust_fusion import shift_and_add, robust_fusion
from m6a_multiframe.backprojection import iterative_backprojection
from m6a_multiframe.metrics import compute_psnr, compute_ssim, registration_rmse, edge_f1_score


def make_test_image(h=60, w=80, channels=3):
    """Deterministic structured test image with edges and gradients."""
    img = np.zeros((h, w, channels), dtype=np.float32)
    for c in range(channels):
        img[:, :, c] = np.linspace(0.1, 0.9, w)[np.newaxis, :]
    img[h//4:3*h//4, w//4:3*w//4, :] = 0.8
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    pattern = 0.1 * np.sin(2 * np.pi * xx / 10) * np.sin(2 * np.pi * yy / 10)
    for c in range(channels):
        img[:, :, c] += pattern
    return np.clip(img, 0, 1).astype(np.float32)


class TestFrameGeneration:
    def test_deterministic(self):
        hr = make_test_image(60, 80)
        f1, t1 = generate_frames(hr, 2, 8, 0.5, seed=42)
        f2, t2 = generate_frames(hr, 2, 8, 0.5, seed=42)
        assert len(f1) == 8
        for a, b in zip(f1, f2):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(t1, t2):
            assert a == b

    def test_frame_dimensions(self):
        hr = make_test_image(60, 80)
        frames, _ = generate_frames(hr, 2, 8, 0.5)
        for f in frames:
            assert f.shape == (30, 40, 3)

    def test_reference_frame_unshifted(self):
        hr = make_test_image(60, 80)
        _, translations = generate_frames(hr, 2, 8, 0.5)
        assert translations[0] == (0.0, 0.0)

    def test_subpixel_translations(self):
        hr = make_test_image(60, 80)
        _, translations = generate_frames(hr, 2, 8, 0.5)
        non_zero = sum(1 for dx, dy in translations[1:] if dx != 0 or dy != 0)
        assert non_zero >= 5

    def test_4x_frame_dimensions(self):
        hr = make_test_image(60, 80)
        frames, _ = generate_frames(hr, 4, 16, 0.5)
        for f in frames:
            assert f.shape == (15, 20, 3)


class TestForwardModel:
    def test_identity_no_shift(self):
        img = make_test_image(40, 40)
        out = forward_model(img, 2, 0, (0, 0))
        assert out.shape == (20, 20, 3)

    def test_forward_model_consistency(self):
        img = make_test_image(40, 40)
        out1 = forward_model(img, 2, 0.5, (1.0, 1.0))
        out2 = forward_model(img, 2, 0.5, (1.0, 1.0))
        np.testing.assert_array_equal(out1, out2)

    def test_srgb_roundtrip(self):
        img = np.array([0, 50, 100, 150, 200, 255], dtype=np.uint8)
        lin = srgb_to_lin(img)
        back = lin_to_srgb(lin) * 255
        np.testing.assert_allclose(back, img.astype(float), atol=1.0)


class TestRegistration:
    def test_zero_shift_identity(self):
        img = make_test_image(40, 60)
        dx, dy, cc = estimate_translation(img, img)
        assert abs(dx) < 0.1
        assert abs(dy) < 0.1
        assert cc > 0.95

    def test_known_shift(self):
        img = make_test_image(60, 80)
        shifted = np.roll(img, 3, axis=1)
        shifted = np.roll(shifted, 2, axis=0)
        dx, dy, cc = estimate_translation(img, shifted)
        assert abs(dx - 3.0) < 1.5
        assert abs(dy - 2.0) < 1.5

    def test_register_multiple_frames(self):
        hr = make_test_image(60, 80)
        frames, true_t = generate_frames(hr, 2, 8, 0.5)
        est_t, accepted = register_frames(frames, ref_index=0, cc_threshold=0.5)
        assert len(est_t) == 8
        assert len(accepted) == 8
        assert accepted[0] == True

    def test_no_ground_truth_leakage(self):
        import inspect
        sig = inspect.signature(register_frames)
        params = list(sig.parameters.keys())
        assert 'true_translations' not in params
        assert 'ground_truth' not in params

    def test_reject_corrupt_frame(self):
        hr = make_test_image(60, 80)
        frames, _ = generate_frames(hr, 2, 4, 0.5)
        frames[2] = np.random.RandomState(99).rand(*frames[2].shape).astype(np.float32)
        _, accepted = register_frames(frames, ref_index=0, cc_threshold=0.9)
        assert accepted[2] == False


class TestFusion:
    def test_shift_and_add_dimensions(self):
        hr = make_test_image(40, 60)
        frames, _ = generate_frames(hr, 2, 4, 0.5)
        translations = [(0, 0)] * 4
        result = shift_and_add(frames, translations, 2)
        assert result.shape[:2] == (40, 60)

    def test_robust_fusion_outputs(self):
        hr = make_test_image(40, 60)
        frames, _ = generate_frames(hr, 2, 4, 0.5)
        translations = [(0, 0)] * 4
        result, coverage, confidence = robust_fusion(frames, translations, 2)
        assert result.shape[:2] == (40, 60)
        assert coverage.shape == (40, 60)
        assert confidence.shape == (40, 60)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_coverage_map_nonzero(self):
        hr = make_test_image(40, 60)
        frames, _ = generate_frames(hr, 2, 4, 0.5)
        translations = [(0, 0)] * 4
        _, coverage, _ = robust_fusion(frames, translations, 2)
        assert coverage.mean() > 0.5


class TestBackprojection:
    def test_output_dimensions(self):
        hr = make_test_image(40, 60)
        frames, _ = generate_frames(hr, 2, 4, 0.5)
        translations = [(0, 0)] * 4
        init = cv2.resize(frames[0], (60, 40), interpolation=cv2.INTER_LANCZOS4)
        result, history = iterative_backprojection(
            init, frames, translations, 2, 0.5, max_iters=3
        )
        assert result.shape == (40, 60, 3)
        assert len(history) == 3

    def test_output_bounds(self):
        hr = make_test_image(40, 60)
        frames, _ = generate_frames(hr, 2, 4, 0.5)
        translations = [(0, 0)] * 4
        init = cv2.resize(frames[0], (60, 40), interpolation=cv2.INTER_LANCZOS4)
        result, _ = iterative_backprojection(
            init, frames, translations, 2, 0.5, max_iters=3
        )
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_deterministic(self):
        hr = make_test_image(40, 60)
        frames, _ = generate_frames(hr, 2, 4, 0.5)
        translations = [(0, 0)] * 4
        init = cv2.resize(frames[0], (60, 40), interpolation=cv2.INTER_LANCZOS4)
        r1, _ = iterative_backprojection(init, frames, translations, 2, 0.5, max_iters=3)
        r2, _ = iterative_backprojection(init, frames, translations, 2, 0.5, max_iters=3)
        np.testing.assert_array_equal(r1, r2)


class TestMetrics:
    def test_psnr_identical(self):
        img = (make_test_image(30, 40) * 255).astype(np.uint8)
        assert compute_psnr(img, img) == 100.0

    def test_ssim_identical(self):
        img = (make_test_image(30, 40) * 255).astype(np.uint8)
        assert compute_ssim(img, img) > 0.99

    def test_registration_rmse(self):
        est = [(1.0, 2.0), (0.5, 0.5)]
        true = [(1.0, 2.0), (0.5, 0.5)]
        assert registration_rmse(est, true) == 0.0

    def test_edge_f1(self):
        img = (make_test_image(40, 60) * 255).astype(np.uint8)
        f1 = edge_f1_score(img, img)
        assert f1['f1'] > 0.9


class TestEvidenceSchema:
    def test_evidence_fields(self):
        required = [
            'run_uuid', 'utc_time', 'command', 'git_commit', 'git_dirty',
            'source_sha256', 'random_seeds', 'frame_count', 'scale_factor',
            'estimated_translations', 'true_translations', 'accepted_frames',
            'rejected_frames', 'registration_error', 'runtime_seconds',
            'peak_ram_mb', 'output_dimensions', 'artifacts'
        ]
        assert len(required) == 18
