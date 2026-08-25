import os
import sys
import uuid
import time
import subprocess
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from minimax_gate.controlled_inputs import generate_controlled_inputs
from minimax_gate.client import generate_image, MissingAPIKeyError
from minimax_gate.evidence import append_evidence
from minimax_gate.configuration import MODEL_ID, API_ENDPOINT, WIDTH, HEIGHT

OUT_DIR = "milestone7a_artifacts"

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        return "unknown"

def main():
    print("=== Milestone 7A: MiniMax Suitability Gate ===")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Ensure controlled inputs
    gt_path = "m4d_test_assets/porche.png"
    inputs_dir = os.path.join(OUT_DIR, "inputs")
    try:
        inputs = generate_controlled_inputs(gt_path, inputs_dir)
        print(f"Controlled inputs generated at {inputs_dir}")
    except FileNotFoundError as e:
        print(str(e))
        return 1

    # Get the latest commit hash (this should be the commit AFTER pushing Step 1)
    commit = get_git_commit()
    print(f"Current commit: {commit}")
    
    # If API key is missing, stop gracefully according to requirements
    if 'MINIMAX_API_KEY' not in os.environ:
        print("\nSTATUS: BLOCKED_MISSING_API_KEY")
        print("Set MINIMAX_API_KEY locally in PowerShell using:")
        print('$env:MINIMAX_API_KEY="your-key"')
        
        # Append minimal evidence to prove we ran the controlled inputs successfully
        record = {
            "run_id": str(uuid.uuid4()),
            "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "BLOCKED_MISSING_API_KEY",
            "commit": commit
        }
        append_evidence(OUT_DIR, record)
        return 0

    print("\nAPI Key found. Beginning generation...")
    # Generation logic will go here when key is provided
    # ...

if __name__ == '__main__':
    sys.exit(main())
