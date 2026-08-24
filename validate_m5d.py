import json
import os
import sys
import subprocess
import traceback

def main():
    try:
        val_res = {
            "pytest": {},
            "viewer_build": {},
            "hashes": {}
        }
        
        # 1. Run Tests
        test_out = subprocess.run([sys.executable, "-m", "pytest", "tests"], capture_output=True, text=True)
        val_res["pytest"]["command"] = "python -m pytest tests"
        val_res["pytest"]["exit_code"] = test_out.returncode
        val_res["pytest"]["stdout"] = test_out.stdout
        val_res["pytest"]["stderr"] = test_out.stderr
        
        if test_out.returncode != 0:
            print(f"Tests failed:\n{test_out.stdout}\n{test_out.stderr}")
            failed = True
        else:
            print("Tests passed.")
            failed = False
            
        # 2. Viewer build
        viewer_dir = "src/viewer"
        build_out = subprocess.run(["npm", "run", "build"], cwd=viewer_dir, capture_output=True, text=True, shell=True)
        val_res["viewer_build"]["command"] = "npm run build"
        val_res["viewer_build"]["exit_code"] = build_out.returncode
        val_res["viewer_build"]["stdout"] = build_out.stdout
        val_res["viewer_build"]["stderr"] = build_out.stderr
        
        if build_out.returncode != 0:
            print(f"Viewer build failed:\n{build_out.stdout}\n{build_out.stderr}")
            failed = True
        else:
            print("Viewer build passed.")
            
        # 3. Check hashes
        if not os.path.exists("m5d_results/evidence.jsonl"):
            print("Missing evidence.jsonl")
            sys.exit(1)
            
        with open("m5d_results/evidence.jsonl", "r") as f:
            lines = f.readlines()
            ev = json.loads(lines[-1])
            
        for a in ev["artifacts"]:
            path = a["path"]
            if not os.path.exists(path):
                print(f"Artifact missing: {path}")
                val_res["hashes"][path] = "missing"
                failed = True
                continue
                
            import hashlib
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if h != a["sha256"]:
                print(f"Hash mismatch for {path}")
                val_res["hashes"][path] = "mismatch"
                failed = True
            else:
                val_res["hashes"][path] = "passed"
                
        with open("m5d_results/validation.json", "w") as f:
            json.dump(val_res, f, indent=2)
            
        if failed:
            sys.exit(1)
            
        print("Validation complete.")
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
