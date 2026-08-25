import os
import sys
content = open('tests/test_m7a.py', 'r').read()
content += """
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
"""
open('tests/test_m7a.py', 'w').write(content)
