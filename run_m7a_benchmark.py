import os
import sys
import argparse
import subprocess
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))
from minimax_gate.controlled_inputs import generate_controlled_inputs

OUT_DIR = "milestone7a_artifacts"

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        return "unknown"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-api", action="store_true", help="Attempt to execute live API generation")
    parser.add_argument("--max-calls", type=int, help="Maximum authorized API calls")
    args = parser.parse_args(argv)

    if args.execute_api:
        print("Milestone 7A API budget exhausted; additional requests are not authorized.")
        sys.exit(1)

    print("=== Milestone 7A: MiniMax Suitability Gate ===")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "generated"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "evaluation_360x220"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "edge_overlays"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "baselines"), exist_ok=True)
    
    gt_path = "m4d_test_assets/porche.png"
    inputs_dir = os.path.join(OUT_DIR, "inputs")
    try:
        inputs = generate_controlled_inputs(gt_path, inputs_dir)
        print(f"Controlled inputs generated at {inputs_dir}")
    except FileNotFoundError as e:
        print(str(e))
        return 1

    commit = get_git_commit()
    print(f"Current commit: {commit}")
    
    for scale, in_path in inputs.items():
        inp = cv2.imread(in_path)
        l_360 = cv2.resize(inp, (360, 220), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(OUT_DIR, "baselines", f"lanczos_{scale}_360x220.png"), l_360)
        l_1440 = cv2.resize(inp, (1440, 880), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(OUT_DIR, "baselines", f"lanczos_{scale}_1440x880.png"), l_1440)
        
    print("\nAPI Execution is DISABLED. Run with --execute-api to attempt generation (which will be rejected by the budget guard).")
    return 0

if __name__ == '__main__':
    sys.exit(main())
