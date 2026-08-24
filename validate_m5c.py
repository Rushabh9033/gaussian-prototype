import os
import sys
import json
import subprocess
import hashlib
import platform

def hash_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    validation_data = {
        "validated_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip(),
        "pytest_command": sys.executable + " -m pytest tests",
        "pytest_exit_code": 1,
        "tests_passed": 0,
        "tests_failed": 0,
        "viewer_build_command": "npm run build",
        "viewer_build_exit_code": 1,
        "dependencies": {
            "python": platform.python_version()
        },
        "hashes": {}
    }
    
    print("Running tests...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.getcwd())
    res_tests = subprocess.run([sys.executable, "-m", "pytest", "tests"], env=env, capture_output=True, text=True)
    validation_data["pytest_exit_code"] = res_tests.returncode
    
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
    
    print("Building viewer...")
    res_build = subprocess.run(["npm", "run", "build"], cwd=os.path.join(os.getcwd(), "src", "viewer"), capture_output=True, text=True, shell=True)
    validation_data["viewer_build_exit_code"] = res_build.returncode
    
    validation_data["hashes"]["results.json"] = hash_file("m5c_results/results.json")
    validation_data["hashes"]["evidence.jsonl"] = hash_file("m5c_results/evidence.jsonl")
    validation_data["hashes"]["provenance.json"] = hash_file("m5c_results/provenance.json")
    
    os.makedirs("m5c_results", exist_ok=True)
    with open("m5c_results/validation.json", "w", newline='\n') as f:
        json.dump(validation_data, f, indent=2)
        f.write('\n')
        
    print("Validation complete.")
    if validation_data["pytest_exit_code"] != 0 or validation_data["viewer_build_exit_code"] != 0:
        print("WARNING: Validation failed! See exit codes.")
        print("Tests exit code:", validation_data["pytest_exit_code"])

if __name__ == "__main__":
    main()
