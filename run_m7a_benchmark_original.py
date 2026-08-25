import os
import sys
import uuid
import time
import subprocess
import traceback
import cv2
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from minimax_gate.controlled_inputs import generate_controlled_inputs
from minimax_gate.client import generate_image, MissingAPIKeyError, VehicleReferenceRejectedError
from minimax_gate.evidence import append_evidence
from minimax_gate.configuration import MODEL_ID, API_ENDPOINT, WIDTH, HEIGHT
from minimax_gate.evaluation import compute_perceptual_hash, hamming_distance, compute_orb_geometric_inlier_ratio
from m6a_multiframe.metrics import compute_psnr, compute_ssim, edge_f1_score
from minimax_gate.call_budget import get_call_count, get_estimated_cost
from minimax_gate.redaction import safe_json_dumps

OUT_DIR = "milestone7a_artifacts"

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        return "unknown"
        
def get_git_status():
    try:
        return subprocess.check_output(["git", "status", "--porcelain"]).strip().decode()
    except Exception:
        return "unknown"

def generate_heatmap(img1, img2):
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
    return heatmap
    
def generate_edge_overlay(img1, img2):
    e1 = cv2.Canny(img1, 100, 200)
    e2 = cv2.Canny(img2, 100, 200)
    overlay = np.zeros_like(img1)
    overlay[e1 > 0] = [255, 0, 0] # Blue for img1 edges
    overlay[e2 > 0] = [0, 0, 255] # Red for img2 edges
    overlap = (e1 > 0) & (e2 > 0)
    overlay[overlap] = [255, 255, 255] # White for matching
    return overlay

def main():
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
    
    if 'MINIMAX_API_KEY' not in os.environ:
        print("\nSTATUS: BLOCKED_MISSING_API_KEY")
        record = {
            "run_id": str(uuid.uuid4()),
            "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "BLOCKED_MISSING_API_KEY",
            "commit": commit
        }
        append_evidence(OUT_DIR, record)
        return 0

    print("\nAPI Key found. Beginning generation...")
    
    gt_img = cv2.imread(gt_path)
    
    # Baselines
    lanczos_imgs = {}
    for scale, in_path in inputs.items():
        inp = cv2.imread(in_path)
        # Baseline 360x220 for metric comparison
        l_360 = cv2.resize(inp, (360, 220), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(OUT_DIR, "baselines", f"lanczos_{scale}_360x220.png"), l_360)
        # Baseline 1440x880 for visual
        l_1440 = cv2.resize(inp, (1440, 880), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(OUT_DIR, "baselines", f"lanczos_{scale}_1440x880.png"), l_1440)
        lanczos_imgs[scale] = l_360

    results = {}
    
    run_id = str(uuid.uuid4())
    
    try:
        for scale in ['2x', '4x']:
            results[scale] = {}
            for seed in [42, 4242]:
                print(f"Generating for {scale} with seed {seed}...")
                
                # Raw GitHub URL pinned to implementation commit e8656d7
                url = f"https://raw.githubusercontent.com/Rushabh9033/gaussian-prototype/64c78d4db38fec1eab953c666a3fa551c22e2135/milestone7a_artifacts/inputs/input_{scale}.png"
                
                t0 = time.time()
                try:
                    img_data, meta = generate_image(url, seed)
                    t1 = time.time()
                    
                    raw_path = os.path.join(OUT_DIR, "generated", f"minimax_{scale}_seed{seed}.png")
                    with open(raw_path, 'wb') as f:
                        f.write(img_data)
                        
                    raw_img = cv2.imread(raw_path)
                    
                    # Downsample to 360x220
                    eval_img = cv2.resize(raw_img, (360, 220), interpolation=cv2.INTER_AREA)
                    eval_path = os.path.join(OUT_DIR, "evaluation_360x220", f"minimax_{scale}_seed{seed}.png")
                    cv2.imwrite(eval_path, eval_img)
                    
                    # Metrics against GT
                    psnr_val = compute_psnr(gt_img, eval_img)
                    ssim_val = compute_ssim(gt_img, eval_img)
                    f1_val = edge_f1_score(gt_img, eval_img)
                    
                    # ORB and Phash
                    inlier_ratio, _ = compute_orb_geometric_inlier_ratio(gt_img, eval_img)
                    h_gt = compute_perceptual_hash(gt_img)
                    h_eval = compute_perceptual_hash(eval_img)
                    phash_dist = int(hamming_distance(h_gt, h_eval)) # cast to int for json serialization
                    
                    results[scale][seed] = {
                        "psnr": float(psnr_val),
                        "ssim": float(ssim_val),
                        "edge_f1": float(f1_val),
                        "orb_inlier_ratio": float(inlier_ratio),
                        "phash_distance": phash_dist,
                        "runtime": t1 - t0,
                        "metadata": meta
                    }
                    
                    # Heatmaps & Overlays
                    hm = generate_heatmap(gt_img, eval_img)
                    cv2.imwrite(os.path.join(OUT_DIR, "heatmaps", f"heatmap_{scale}_seed{seed}.png"), hm)
                    
                    eo = generate_edge_overlay(gt_img, eval_img)
                    cv2.imwrite(os.path.join(OUT_DIR, "edge_overlays", f"edge_overlay_{scale}_seed{seed}.png"), eo)
                    
                except VehicleReferenceRejectedError:
                    print(f"Vehicle reference rejected for {scale} seed {seed}")
                    results[scale][seed] = {"error": "API_NOT_SUITABLE", "reason": "Vehicle reference rejected"}
                except Exception as e:
                    print(f"Error during generation: {e}")
                    results[scale][seed] = {"error": str(e), "traceback": traceback.format_exc()}
                    
        # Evidence record
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            f.write(safe_json_dumps(results))
            
        record = {
            "run_id": run_id,
            "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "COMPLETED",
            "commit": commit,
            "git_status": get_git_status(),
            "calls_made": get_call_count(),
            "cost": get_estimated_cost(),
            "results": results
        }
        append_evidence(OUT_DIR, record)
        print("Generation complete. Results saved.")
        
    except Exception as e:
        record = {
            "run_id": run_id,
            "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "FAILED",
            "commit": commit,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        append_evidence(OUT_DIR, record)
        print(f"Run failed: {e}")
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
