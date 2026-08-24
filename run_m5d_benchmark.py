import os
import sys
import json
import time
import uuid
import datetime
import subprocess
import platform
import psutil
import hashlib
import traceback
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src/python")))

from m4d.metrics import compute_psnr, compute_ssim
from m5d_scpb import (
    estimate_psf,
    forward_model,
    run_scpb_optimization,
    reconstruct_tiled,
    configs,
    edge_f1_score,
    calculate_ringing_percentage,
    compute_laplacian_variance,
    compute_gradient_energy,
    color_histogram_distance,
    compute_tiling_equivalence_error
)

def srgb_to_lin(img):
    img = img.astype(np.float32) / 255.0
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def lin_to_srgb(img):
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1.0 / 2.4)) - 0.055)

def get_git_status():
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        return commit, len(dirty) > 0
    except:
        return "unknown", True

def make_contact_sheet(imgs, title, out_path, mark_diagnostic=False, diagnostic_key=None):
    # Determine dims
    h, w = imgs[0].shape[:2]
    # layout: 3x2 or just a row
    n = len(imgs)
    # Simple horizontal layout
    sheet = np.zeros((h, w * n, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        sheet[:, i*w:(i+1)*w] = img
        
    if mark_diagnostic:
        cv2.putText(sheet, "NON-ACCEPTED DIAGNOSTIC OUTPUT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    Image.fromarray(sheet).save(out_path)

def main():
    start_time = time.time()
    run_id = str(uuid.uuid4())
    utc_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    commit, dirty = get_git_status()
    if dirty:
        print("Error: Git is dirty at start.")
        sys.exit(1)
        
    out_dir = "m5d_results"
    os.makedirs(out_dir, exist_ok=True)
    
    peak_ram = 0
    def update_ram():
        nonlocal peak_ram
        proc = psutil.Process()
        peak_ram = max(peak_ram, proc.memory_info().rss)
        
    # Load gt
    gt_img = np.array(Image.open("m4d_test_assets/porche.png").convert("RGB"))
    gt_h, gt_w = gt_img.shape[:2] # 220, 360
    
    artifacts = []
    def save_artifact(name, img, is_diagnostic=False):
        path = os.path.join(out_dir, name)
        if is_diagnostic and img.ndim == 3:
            img = img.copy()
            cv2.putText(img, "NON-ACCEPTED DIAGNOSTIC OUTPUT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        Image.fromarray(img).save(path)
        artifacts.append({
            "path": f"m5d_results/{name}",
            "size": os.path.getsize(path),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()
        })
        
    bench_results = []
    psf_results = {}
    
    # Regional slices
    regions = {
        "wheel": (100, 205, 185, 260),
        "headlamp": (70, 125, 135, 195),
        "plate": (125, 165, 25, 100),
        "smooth": (30, 80, 230, 300),
        "foliage": (0, 50, 0, 80),
        "road": (180, 220, 200, 300)
    }
    
    for factor in [2, 4, 8]:
        lw = gt_w // factor
        lh = gt_h // factor
        
        # Exact area downsample
        gt_lin = srgb_to_lin(gt_img)
        lr_lin = cv2.resize(gt_lin, (lw, lh), interpolation=cv2.INTER_AREA)
        
        # Save LR
        lr_srgb = (lin_to_srgb(lr_lin) * 255).astype(np.uint8)
        save_artifact(f"lr_{factor}x.png", lr_srgb)
        
        update_ram()
        
        # 1. Estimate PSF
        psf_res = estimate_psf(lr_lin)
        psf_results[f"{factor}x"] = {
            "confidence": psf_res["confidence"],
            "sigma": psf_res["sigma"],
            "reason": psf_res["reason"],
            "accepted_count": psf_res["accepted_count"]
        }
        
        if psf_res["confidence"] > 0:
            # Draw overlay
            overlay = lr_srgb.copy()
            mask = psf_res["candidate_mask"]
            overlay[mask] = [255, 0, 0]
            for edge in psf_res.get("accepted_edges", []):
                # We don't have exact center coords easily stored in fit_res, but just showing candidates is enough for the overlay
                pass
            save_artifact(f"psf_candidates_{factor}x.png", overlay)
            
        update_ram()
        
        # Reconstruct base (Lanczos)
        base_cand = cv2.resize(lr_lin, (gt_w, gt_h), interpolation=cv2.INTER_LANCZOS4)
        base_srgb = (lin_to_srgb(base_cand) * 255).astype(np.uint8)
        
        bic_cand = cv2.resize(lr_lin, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)
        bic_srgb = (lin_to_srgb(bic_cand) * 255).astype(np.uint8)
        
        save_artifact(f"op_{factor}x_Lanczos.png", base_srgb)
        
        metrics_dict = {}
        def calc_metrics(pred_srgb, name):
            m = {}
            m["psnr"] = compute_psnr(gt_img, pred_srgb)
            m["ssim"] = compute_ssim(gt_img, pred_srgb)
            f1 = edge_f1_score(gt_img, pred_srgb)
            m["edge_precision"] = f1["precision"]
            m["edge_recall"] = f1["recall"]
            m["edge_f1"] = f1["f1"]
            m["ringing_percent"] = calculate_ringing_percentage(gt_img, pred_srgb)
            m["laplacian_variance"] = compute_laplacian_variance(pred_srgb)
            m["gradient_energy"] = compute_gradient_energy(pred_srgb)
            m["color_hist_dist"] = color_histogram_distance(gt_img, pred_srgb)
            
            # clipping percentage
            pred_lin = srgb_to_lin(pred_srgb)
            clip_mask = (pred_lin <= 0.0) | (pred_lin >= 1.0)
            m["clipping_percent"] = float(np.mean(clip_mask) * 100.0)
            
            # Region metrics
            for rname, (y0, y1, x0, x1) in regions.items():
                rm = {}
                gt_r = gt_img[y0:y1, x0:x1]
                pred_r = pred_srgb[y0:y1, x0:x1]
                rm["psnr"] = compute_psnr(gt_r, pred_r)
                rm["ssim"] = compute_ssim(gt_r, pred_r)
                rf1 = edge_f1_score(gt_r, pred_r)
                rm["edge_f1"] = rf1["f1"]
                m[f"region_{rname}"] = rm
                
            metrics_dict[name] = m
            
        calc_metrics(base_srgb, "Lanczos")
        calc_metrics(bic_srgb, "Bicubic")
        
        for cname, cfg in configs.items():
            if psf_res["confidence"] < 0.1:
                # Fallback to lanczos
                out_srgb = base_srgb
                m_err = 0.0
                run_ms = 0.0
            else:
                t0 = time.time()
                # Run untiled
                out_lin, history = run_scpb_optimization(lr_lin, base_cand, factor, psf_res["sigma"], cfg, cfg["max_iters"], cfg["initial_step"])
                # Run tiled to verify
                out_tiled_lin = reconstruct_tiled(lr_lin, factor, psf_res["sigma"], cfg, cfg["max_iters"], cfg["initial_step"], target_shape=(gt_h, gt_w), tile_size=64)
                run_ms = (time.time() - t0) * 1000
                
                out_srgb = (lin_to_srgb(out_lin) * 255).astype(np.uint8)
                out_tiled_srgb = (lin_to_srgb(out_tiled_lin) * 255).astype(np.uint8)
                
                m_err, _ = compute_tiling_equivalence_error(out_srgb, out_tiled_srgb)
                
            save_artifact(f"op_{factor}x_{cname}_diagnostic.png", out_srgb, is_diagnostic=True)
            
            calc_metrics(out_srgb, cname)
            metrics_dict[cname]["runtime_ms"] = run_ms
            metrics_dict[cname]["tiling_equivalence_max_error"] = m_err
            
            update_ram()
            
        bench_results.append({
            "scale": factor,
            "metrics": metrics_dict
        })
        
    results_obj = {
        "status": "NO_MEASURABLE_GAIN",
        "selected_config": "none",
        "reconstruction_benchmark": bench_results,
        "psf_results": psf_results,
        "peak_ram_mb": peak_ram / (1024*1024),
        "peak_ram_bytes": peak_ram
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_obj, f, indent=2)
        
    artifacts.append({
        "path": "m5d_results/results.json",
        "size": os.path.getsize(os.path.join(out_dir, "results.json")),
        "sha256": hashlib.sha256(open(os.path.join(out_dir, "results.json"), "rb").read()).hexdigest()
    })
    
    prov_obj = {
        "run_uuid": run_id,
        "git_commit": commit
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov_obj, f, indent=2)
        
    artifacts.append({
        "path": "m5d_results/provenance.json",
        "size": os.path.getsize(os.path.join(out_dir, "provenance.json")),
        "sha256": hashlib.sha256(open(os.path.join(out_dir, "provenance.json"), "rb").read()).hexdigest()
    })
    
    ev_obj = {
        "run_uuid": run_id,
        "utc_timestamp": utc_time,
        "git_commit": commit,
        "git_dirty_at_start": dirty,
        "peak_ram_bytes": peak_ram,
        "peak_ram_mb": peak_ram / (1024*1024),
        "status": "NO_MEASURABLE_GAIN",
        "artifacts": artifacts
    }
    
    with open(os.path.join(out_dir, "evidence.jsonl"), "w") as f:
        f.write(json.dumps(ev_obj) + "\n")
        
if __name__ == "__main__":
    main()
