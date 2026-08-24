import json
import os
import sys
import subprocess
import traceback

def main():
    try:
        # 1. Run Tests
        test_out = subprocess.run([sys.executable, "-m", "pytest", "tests/test_m5d.py"], capture_output=True, text=True)
        if test_out.returncode != 0:
            print(f"Tests failed:\n{test_out.stdout}\n{test_out.stderr}")
            sys.exit(1)
            
        print("Tests passed.")
        
        # 2. Check hashes
        if not os.path.exists("m5d_results/evidence.jsonl"):
            print("Missing evidence.jsonl")
            sys.exit(1)
            
        with open("m5d_results/evidence.jsonl", "r") as f:
            ev = json.loads(f.readline())
            
        for a in ev["artifacts"]:
            path = a["path"]
            if not os.path.exists(path):
                print(f"Artifact missing: {path}")
                sys.exit(1)
                
            import hashlib
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if h != a["sha256"]:
                print(f"Hash mismatch for {path}")
                sys.exit(1)
                
        print("Validation complete.")
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
