import sys
import os
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from minimax_gate.configuration import PROMPT, WIDTH, HEIGHT, MODEL_ID, API_ENDPOINT
from minimax_gate.call_budget import check_budget, reset_budget, get_call_count, get_estimated_cost
from minimax_gate.redaction import redact_secrets, safe_json_dumps
from minimax_gate.client import generate_image, MissingAPIKeyError, APIError, VehicleReferenceRejectedError
from minimax_gate.controlled_inputs import generate_controlled_inputs
from minimax_gate.evaluation import compute_perceptual_hash, hamming_distance, compute_orb_geometric_inlier_ratio
import run_m7a_benchmark
import validate_m7a

class TestMiniMaxGate:
    def setup_method(self):
        reset_budget()
        self.original_key = os.environ.get('MINIMAX_API_KEY')
        os.environ['MINIMAX_API_KEY'] = 'test-secret-key-123'
        
    def teardown_method(self):
        if self.original_key is None:
            os.environ.pop('MINIMAX_API_KEY', None)
        else:
            os.environ['MINIMAX_API_KEY'] = self.original_key

    def test_redaction(self):
        data = {"error": "Failed with test-secret-key-123 at endpoint"}
        redacted = redact_secrets(data, "test-secret-key-123")
        assert "test-secret-key-123" not in redacted["error"]
        assert "***REDACTED***" in redacted["error"]
        
    def test_budget_enforcement(self):
        for _ in range(4):
            check_budget()
        with pytest.raises(RuntimeError, match="API budget exceeded"):
            check_budget()
            
    def test_missing_api_key(self):
        del os.environ['MINIMAX_API_KEY']
        with pytest.raises(MissingAPIKeyError):
            generate_image("http://fake", 42)
            
    def test_cost_calculation(self):
        check_budget()
        check_budget()
        assert get_call_count() == 2
        assert get_estimated_cost() == 0.007
        
    def test_controlled_inputs(self, tmpdir):
        gt_path = os.path.join(tmpdir, "gt.png")
        cv2.imwrite(gt_path, np.zeros((220, 360, 3), dtype=np.uint8))
        paths = generate_controlled_inputs(gt_path, str(tmpdir))
        
        in2 = cv2.imread(paths['2x'])
        assert in2.shape == (110, 180, 3)
        
        in4 = cv2.imread(paths['4x'])
        assert in4.shape == (55, 90, 3)

    def test_evaluation_orb(self):
        img_path = os.path.join(os.path.dirname(__file__), '..', 'm4d_test_assets', 'porche.png')
        img1 = cv2.imread(img_path)
        img1 = cv2.resize(img1, (360, 220))
        
        img2 = img1.copy()
        ratio, matches = compute_orb_geometric_inlier_ratio(img1, img2)
        assert matches > 5, f"Meaningful match count expected, got {matches}"
        assert ratio > 0.9, f"Strong geometric inlier ratio for identical images expected, got {ratio}"
        
        # Transformed image
        img3 = np.zeros_like(img1)
        ratio_transformed, matches_transformed = compute_orb_geometric_inlier_ratio(img1, img3)
        assert ratio_transformed < 0.8, "Lower agreement for an unrelated image expected"
        
    def test_perceptual_hash(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[50:, :] = 255
        h1 = compute_perceptual_hash(img)
        h2 = compute_perceptual_hash(img)
        assert hamming_distance(h1, h2) == 0

    def test_removed_5xx_retry_submits_exactly_one_mocked_http_request(self, monkeypatch):
        import requests
        class MockResp:
            status_code = 500
            text = "Internal Server Error"
            def raise_for_status(self):
                raise requests.exceptions.HTTPError("500")
                
        call_count = [0]
        def mock_post(*args, **kwargs):
            call_count[0] += 1
            return MockResp()
            
        monkeypatch.setattr(requests, "post", mock_post)
        
        with pytest.raises(APIError):
            generate_image("http://fake", 42)
            
        assert call_count[0] == 1, "Expected exactly one mocked HTTP request on 5xx error"
        
    def test_closure_guard_submits_zero_http_requests(self, monkeypatch):
        def mock_post(*args, **kwargs):
            raise RuntimeError("HTTP REQUEST MADE")
        import requests
        monkeypatch.setattr(requests, "post", mock_post)
        
        with pytest.raises(SystemExit) as e:
            run_m7a_benchmark.main(["--execute-api", "--max-calls", "4"])
        assert e.value.code == 1

    def test_default_execution_submits_zero_http_requests(self, monkeypatch):
        def mock_post(*args, **kwargs):
            raise RuntimeError("HTTP REQUEST MADE")
        import requests
        monkeypatch.setattr(requests, "post", mock_post)
        
        code = run_m7a_benchmark.main([])
        assert code == 0
        
    def test_main_does_not_depend_on_global_sys_argv(self, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "argv", ["fake_script.py", "--invalid-flag"])
        # Should parse the passed argv, not sys.argv
        code = run_m7a_benchmark.main([])
        assert code == 0

    def test_validator_accepts_honest_report(self):
        assert validate_m7a.validate_m7a() == 0
        
    def test_validator_rejects_actual_final_status_line_containing_api_not_suitable(self, tmpdir, monkeypatch):
        # Temporarily mock the report file read in validate_m7a
        original_open = open
        def mock_open(path, *args, **kwargs):
            if "MILESTONE7A_REPORT.md" in str(path):
                import io
                return io.StringIO("**Final Status:** `API_NOT_SUITABLE`\n16 historical attempts\n0 successful images\nCost:** Unknown")
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr("builtins.open", mock_open)
        assert validate_m7a.validate_m7a() == 1

    def test_validator_confirms_exactly_16_archived_attempts(self, monkeypatch):
        original_open = open
        def mock_open(path, *args, **kwargs):
            if "MILESTONE7A_REPORT.md" in str(path):
                import io
                # Omit "16 historical attempts"
                return io.StringIO("**Final Status:** `BLOCKED_API_ACCESS`\n0 successful images\nCost:** Unknown")
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr("builtins.open", mock_open)
        assert validate_m7a.validate_m7a() == 1

    def test_validator_confirms_zero_successful_images(self, monkeypatch):
        original_open = open
        def mock_open(path, *args, **kwargs):
            if "MILESTONE7A_REPORT.md" in str(path):
                import io
                # Omit "0 successful images"
                return io.StringIO("**Final Status:** `BLOCKED_API_ACCESS`\n16 historical attempts\nCost:** Unknown")
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr("builtins.open", mock_open)
        assert validate_m7a.validate_m7a() == 1
        
    def test_tampered_legacy_evidence_fails(self, tmpdir, monkeypatch):
        import hashlib
        def mock_sha256(data):
            class FakeHash:
                def hexdigest(self):
                    return "badhash"
            return FakeHash()
        monkeypatch.setattr(hashlib, "sha256", mock_sha256)
        assert validate_m7a.validate_m7a() == 1
        
    def test_temporary_helper_files_are_absent(self):
        assert not os.path.exists("append_test.py")
        assert not os.path.exists("patch_m7a.py")
        assert not os.path.exists("run_m7a_benchmark_original.py")

    def test_previous_m5c_and_m6a_files_still_match_commit_121533d(self):
        import subprocess
        # verify no diff against 121533d for specific files
        out = subprocess.check_output(["git", "diff", "121533db4c4f353620d9b99fc586d15d959c67e1", "--", "m6a_results/validation.json", "tests/test_m5c.py"])
        assert len(out.strip()) == 0
