import os
import sys
import json
import uuid
import datetime
import subprocess
import platform
import hashlib

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def hash_file(path: str) -> str:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    validation_data = {
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_commit_validated": get_git_commit(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python_version": platform.python_version()
    }

    # 1. Run Tests
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    # We use python -m pytest to ensure it runs correctly on this OS
    pytest_cmd = [sys.executable, "-m", "pytest", "tests"]
    validation_data["pytest_command"] = " ".join(pytest_cmd)
    
    print("Running tests...")
    res_tests = subprocess.run(pytest_cmd, env=env, capture_output=True, text=True)
    validation_data["pytest_exit_code"] = res_tests.returncode
    
    # Try to parse passed/failed from output
    passed = 0
    failed = 0
    for line in res_tests.stdout.splitlines():
        if " passed" in line and "==" in line:
            import re
            m = re.search(r'(\d+) passed', line)
            if m: passed = int(m.group(1))
            m = re.search(r'(\d+) failed', line)
            if m: failed = int(m.group(1))
            break
    
    validation_data["tests_passed"] = passed
    validation_data["tests_failed"] = failed

    # 2. Viewer build
    viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "viewer")
    viewer_cmd = ["npm", "run", "build"]
    validation_data["viewer_build_command"] = " ".join(viewer_cmd)
    
    print("Building viewer...")
    res_viewer = subprocess.run(viewer_cmd, cwd=viewer_dir, capture_output=True, text=True, shell=True)
    validation_data["viewer_build_exit_code"] = res_viewer.returncode

    # 3. Hashes
    results_path = os.path.join("m5_results", "results.json")
    evidence_path = os.path.join("m5_results", "evidence.jsonl")
    package_path = os.path.join("m5_results", "vrc_package.zip")
    
    validation_data["hashes"] = {
        "results.json": hash_file(results_path),
        "evidence.jsonl": hash_file(evidence_path),
        "vrc_package.zip": hash_file(package_path)
    }

    os.makedirs("m5_results", exist_ok=True)
    with open(os.path.join("m5_results", "validation.json"), "w", newline='\n') as f:
        json.dump(validation_data, f, indent=2)

    print("Validation complete.")

if __name__ == "__main__":
    main()
