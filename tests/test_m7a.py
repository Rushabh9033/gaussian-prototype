import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from minimax_gate.configuration import PROMPT, WIDTH, HEIGHT, MODEL_ID, API_ENDPOINT
from minimax_gate.call_budget import check_budget, reset_budget, get_call_count, get_estimated_cost
from minimax_gate.redaction import redact_secrets, safe_json_dumps
from minimax_gate.client import generate_image, MissingAPIKeyError, APIError, VehicleReferenceRejectedError
from minimax_gate.controlled_inputs import generate_controlled_inputs
from minimax_gate.evaluation import compute_perceptual_hash, hamming_distance, compute_orb_geometric_inlier_ratio

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
        import cv2
        cv2.imwrite(gt_path, np.zeros((220, 360, 3), dtype=np.uint8))
        paths = generate_controlled_inputs(gt_path, str(tmpdir))
        
        in2 = cv2.imread(paths['2x'])
        assert in2.shape == (110, 180, 3)
        
        in4 = cv2.imread(paths['4x'])
        assert in4.shape == (55, 90, 3)

    def test_evaluation_orb(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        cv2.rectangle(img1, (20, 20), (80, 80), (255, 255, 255), -1)
        cv2.rectangle(img2, (20, 20), (80, 80), (255, 255, 255), -1)
        ratio, matches = compute_orb_geometric_inlier_ratio(img1, img2)
        assert ratio >= 0.0
        
    def test_perceptual_hash(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[50:, :] = 255
        h1 = compute_perceptual_hash(img)
        h2 = compute_perceptual_hash(img)
        assert hamming_distance(h1, h2) == 0
        
    def test_viewer_does_not_use_api(self):
        viewer_main = os.path.join(os.path.dirname(__file__), '..', 'src', 'viewer', 'src', 'main.ts')
        if os.path.exists(viewer_main):
            with open(viewer_main, 'r') as f:
                content = f.read()
                assert "api.minimax.io" not in content

    def test_explicit_execute_api_enforcement(self):
        import subprocess
        res = subprocess.run([sys.executable, "run_m7a_benchmark.py"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "API Execution is DISABLED" in res.stdout
        
    def test_permanent_exhausted_budget_guard(self):
        import subprocess
        res = subprocess.run([sys.executable, "run_m7a_benchmark.py", "--execute-api", "--max-calls", "4"], capture_output=True, text=True)
        assert res.returncode == 1
        assert "Milestone 7A API budget exhausted" in res.stdout
        
    def test_zero_http_requests_when_guarded(self, monkeypatch):
        def mock_post(*args, **kwargs):
            raise RuntimeError("HTTP REQUEST MADE")
        import requests
        monkeypatch.setattr(requests, "post", mock_post)
        
        from run_m7a_benchmark import main
        import argparse
        sys.argv = ["run_m7a_benchmark.py", "--execute-api", "--max-calls", "4"]
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1

    def test_schema_array_parsing_and_base64(self, monkeypatch):
        import requests
        from minimax_gate.client import generate_image
        import base64
        import cv2
        class MockResp:
            status_code = 200
            def json(self):
                img = np.zeros((880, 1440, 3), dtype=np.uint8)
                _, buf = cv2.imencode('.png', img)
                b64 = base64.b64encode(buf).decode('utf-8')
                return {
                    "base_resp": {"status_code": 0},
                    "metadata": {"success_count": "1"},
                    "data": {"image_base64": [b64]}
                }
            def raise_for_status(self): pass
        monkeypatch.setattr(requests, "post", lambda *a, **k: MockResp())
        os.environ['MINIMAX_API_KEY'] = 'test-secret-key-123'
        img_bytes, meta = generate_image("http://fake", 42)
        assert isinstance(img_bytes, bytes)

    def test_quota_error_stops_immediately(self, monkeypatch):
        import requests
        from minimax_gate.client import generate_image, QuotaError
        class MockResp:
            status_code = 200
            def json(self):
                return {
                    "base_resp": {"status_code": 2056, "status_msg": "Token Limit"}
                }
            def raise_for_status(self): pass
        monkeypatch.setattr(requests, "post", lambda *a, **k: MockResp())
        os.environ['MINIMAX_API_KEY'] = 'test-secret-key-123'
        with pytest.raises(QuotaError, match="Quota exceeded: Token Limit"):
            generate_image("http://fake", 42)

    def test_dirty_tree_rejection(self):
        import subprocess
        def fake_git_status(): return "M some_file.py"
        import run_m7a_benchmark
        orig = run_m7a_benchmark.get_git_status
        run_m7a_benchmark.get_git_status = fake_git_status
        sys.argv = ["run_m7a_benchmark.py", "--execute-api", "--max-calls", "4"]
        os.environ['MINIMAX_API_KEY'] = 'test-secret-key-123'
        with pytest.raises(SystemExit) as e:
            run_m7a_benchmark.main()
        assert e.value.code == 1
        run_m7a_benchmark.get_git_status = orig
