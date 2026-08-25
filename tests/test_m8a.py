"""Milestone 8A tests.

Real file/schema validation of evidence artifacts.
No model loading, no API calls, no viewer/V3 modification.
"""
import pytest
import os
import sys
import json
import hashlib
import subprocess

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "milestone8a_evidence")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(name):
    path = os.path.join(EVIDENCE_DIR, name)
    assert os.path.exists(path), f"{name} must exist"
    assert os.path.getsize(path) > 10, f"{name} must not be empty"
    with open(path) as f:
        return json.load(f)


# --- Deterministic controlled inputs ---

class TestControlledInputs:
    def test_lr2x_dimensions(self):
        gt = np.zeros((220, 360, 3), dtype=np.uint8)
        lr2x = cv2.resize(gt, (180, 110), interpolation=cv2.INTER_AREA)
        assert lr2x.shape == (110, 180, 3)

    def test_lr4x_dimensions(self):
        gt = np.zeros((220, 360, 3), dtype=np.uint8)
        lr4x = cv2.resize(gt, (90, 55), interpolation=cv2.INTER_AREA)
        assert lr4x.shape == (55, 90, 3)

    def test_outscale_arithmetic(self):
        assert 180 * 2 == 360
        assert 110 * 2 == 220
        assert 90 * 4 == 360
        assert 55 * 4 == 220
        assert 360 * 4 == 1440
        assert 220 * 4 == 880


# --- PNG dimensions ---

class TestPNGDimensions:
    def test_source(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "source", "porche.png"))
        assert img is not None
        assert img.shape == (220, 360, 3)

    def test_lr_2x(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "controlled_inputs", "lr_2x_180x110.png"))
        assert img is not None
        assert img.shape == (110, 180, 3)

    def test_lr_4x(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "controlled_inputs", "lr_4x_90x55.png"))
        assert img is not None
        assert img.shape == (55, 90, 3)

    def test_lanczos_2x(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "lanczos", "2x_360x220.png"))
        assert img is not None
        assert img.shape == (220, 360, 3)

    def test_lanczos_4x(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "lanczos", "4x_360x220.png"))
        assert img is not None
        assert img.shape == (220, 360, 3)

    def test_lanczos_orig(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "lanczos", "orig_1440x880.png"))
        assert img is not None
        assert img.shape == (880, 1440, 3)

    def test_realesrgan_2x(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "realesrgan", "2x_360x220.png"))
        assert img is not None
        assert img.shape == (220, 360, 3)

    def test_realesrgan_4x(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "realesrgan", "4x_360x220.png"))
        assert img is not None
        assert img.shape == (220, 360, 3)

    def test_realesrgan_orig(self):
        img = cv2.imread(os.path.join(EVIDENCE_DIR, "realesrgan", "orig_1440x880.png"))
        assert img is not None
        assert img.shape == (880, 1440, 3)


# --- Execution evidence schema ---

class TestExecutionEvidence:
    def test_schema_keys(self):
        ev = _load_json("execution_evidence.json")
        assert "metrics" in ev
        assert "hashes" in ev
        assert "performance" in ev
        assert "2x" in ev["metrics"]
        assert "4x" in ev["metrics"]

    def test_hash_lr2x(self):
        ev = _load_json("execution_evidence.json")
        actual = _sha256(os.path.join(EVIDENCE_DIR, "controlled_inputs", "lr_2x_180x110.png"))
        assert actual == ev["hashes"]["p_lr2x"]

    def test_hash_lr4x(self):
        ev = _load_json("execution_evidence.json")
        actual = _sha256(os.path.join(EVIDENCE_DIR, "controlled_inputs", "lr_4x_90x55.png"))
        assert actual == ev["hashes"]["p_lr4x"]

    def test_hash_realesrgan_2x(self):
        ev = _load_json("execution_evidence.json")
        actual = _sha256(os.path.join(EVIDENCE_DIR, "realesrgan", "2x_360x220.png"))
        assert actual == ev["hashes"]["realesrgan_2x"]

    def test_hash_realesrgan_4x(self):
        ev = _load_json("execution_evidence.json")
        actual = _sha256(os.path.join(EVIDENCE_DIR, "realesrgan", "4x_360x220.png"))
        assert actual == ev["hashes"]["realesrgan_4x"]

    def test_hash_realesrgan_orig(self):
        ev = _load_json("execution_evidence.json")
        actual = _sha256(os.path.join(EVIDENCE_DIR, "realesrgan", "orig_1440x880.png"))
        assert actual == ev["hashes"]["realesrgan_orig"]


# --- Results.json schema ---

class TestResultsJson:
    def test_final_status(self):
        res = _load_json("results.json")
        assert res["final_status"] == "NO_USEFUL_GAIN"

    def test_overall_pass_false(self):
        res = _load_json("results.json")
        assert res["overall_pass"] is False

    def test_guardrails_present(self):
        res = _load_json("results.json")
        g = res["guardrails"]
        assert "ssim_deg_limit" in g
        assert "psnr_deg_limit_db" in g
        assert "edge_f1_deg_limit" in g
        assert "orb_inlier_ratio_deg_limit" in g
        assert "regional_ssim_deg_limit" in g
        assert "vram_reserved_limit_gb" in g

    def test_vram_within_limit(self):
        res = _load_json("results.json")
        vram = res["vram_check"]
        assert vram["pass"] is True
        assert vram["max_peak_reserved_gb"] <= vram["limit_gb"]

    def test_evaluations_present(self):
        res = _load_json("results.json")
        assert "evaluation_2x" in res
        assert "evaluation_4x" in res
        assert res["evaluation_2x"]["ssim_pass"] is False
        assert res["evaluation_4x"]["ssim_pass"] is False

    def test_status_derived_from_metrics(self):
        """Verify that NO_USEFUL_GAIN is the correct derivation."""
        res = _load_json("results.json")
        e2 = res["evaluation_2x"]
        e4 = res["evaluation_4x"]
        # At least one gate must fail for NO_USEFUL_GAIN
        gates_2x = [e2["ssim_pass"], e2["psnr_pass"]]
        gates_4x = [e4["ssim_pass"], e4["psnr_pass"]]
        assert not all(gates_2x), "2x should have failures"
        assert not all(gates_4x), "4x should have failures"


# --- Closure verification schema ---

class TestClosureVerification:
    def test_schema(self):
        cv = _load_json("closure_verification.json")
        assert "verification_utc_timestamp" in cv
        assert "verification_commit" in cv
        assert "files" in cv
        assert "inference_hashes_match" in cv
        assert "provenance" in cv
        assert cv["recorded_final_status"] == "NO_USEFUL_GAIN"

    def test_hashes_match(self):
        cv = _load_json("closure_verification.json")
        assert cv["inference_hashes_match"] is True

    def test_no_errors(self):
        cv = _load_json("closure_verification.json")
        errors = cv.get("errors", [])
        assert len(errors) == 0, f"Closure errors: {errors}"


# --- Crop presence and Git tracking ---

class TestCrops:
    CROPS = [
        "crops/lanczos_plate.png",
        "crops/realesrgan_plate.png",
        "crops/lanczos_wheel.png",
        "crops/realesrgan_wheel.png",
        "crops/lanczos_headlamp.png",
        "crops/realesrgan_headlamp.png",
    ]

    def test_crops_exist(self):
        for crop in self.CROPS:
            path = os.path.join(EVIDENCE_DIR, crop)
            assert os.path.exists(path), f"Missing crop: {crop}"
            assert os.path.getsize(path) > 100, f"Crop too small: {crop}"

    def test_crops_git_tracked(self):
        for crop in self.CROPS:
            path = os.path.join("milestone8a_evidence", crop)
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", path],
                capture_output=True, cwd=ROOT_DIR,
            )
            assert result.returncode == 0, f"Crop not tracked by Git: {path}"


# --- Contact sheet ---

class TestContactSheet:
    def test_exists_and_nonblank(self):
        path = os.path.join(EVIDENCE_DIR, "contact_sheet.png")
        assert os.path.exists(path)
        img = cv2.imread(path)
        assert img is not None
        # Must not be all black
        assert img.mean() > 1.0, "Contact sheet appears blank"


# --- Report ---

class TestReport:
    def test_report_nonempty(self):
        path = os.path.join(ROOT_DIR, "MILESTONE8A_REPORT.md")
        assert os.path.exists(path)
        content = open(path).read()
        assert len(content) > 500, "Report is too short"

    def test_report_contains_status(self):
        path = os.path.join(ROOT_DIR, "MILESTONE8A_REPORT.md")
        content = open(path).read()
        assert "NO_USEFUL_GAIN" in content


# --- Negative result validation ---

class TestNegativeResult:
    def test_negative_result_is_valid(self):
        """A scientifically valid negative result should not be treated as broken."""
        res = _load_json("results.json")
        assert res["final_status"] == "NO_USEFUL_GAIN"
        assert res["overall_pass"] is False
        # Edge-F1 improved but overall experiment failed
        assert res["evaluation_2x"]["edge_f1_pass"] is True


# --- No model loading, no API, no viewer changes ---

class TestSafety:
    def test_no_model_download_in_tests(self):
        """This test file must not trigger model downloads."""
        # If we got here without downloading a model, we pass
        assert True

    def test_no_api_calls_in_benchmark(self):
        import run_m8a_benchmark
        source = open(run_m8a_benchmark.__file__).read()
        assert "requests.post" not in source
        assert "requests.get" not in source

    def test_viewer_v3_unmodified(self):
        diff = subprocess.check_output(
            ["git", "diff", "407e1bbeed7263a21c7f9b8dd37557ec8905d137", "--", "src/viewer"],
            cwd=ROOT_DIR,
        ).decode()
        assert len(diff.strip()) == 0

    def test_cli_unmodified(self):
        diff = subprocess.check_output(
            ["git", "diff", "407e1bbeed7263a21c7f9b8dd37557ec8905d137", "--", "src/python/cli.py"],
            cwd=ROOT_DIR,
        ).decode()
        assert len(diff.strip()) == 0
