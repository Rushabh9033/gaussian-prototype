import os
import sys
import json
import time
import uuid
import datetime
import psutil
import subprocess
import hashlib
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath("src/python"))
from m5d_scpb import estimate_psf, run_scpb_optimization, configs
from m5d_scpb.tiling import reconstruct_tiled
from m5d_scpb.forward_model import forward_model
from m5d_scpb.metrics import (
    compute_psnr, compute_ssim, edge_f1_score, calculate_ringing_percentage,
    compute_laplacian_variance, compute_gradient_energy, color_histogram_distance,
    compute_tiling_equivalence_error
)

def get_git_status():
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "-uno"]).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        return commit, bool(dirty)
    except:
        return "unknown", True

def lin_to_srgb(img):
    img = np.clip(img, 0, 1)
    mask = img <= 0.0031308
    out = np.empty_like(img)
    out[mask] = img[mask] * 12.92
    out[~mask] = 1.055 * (img[~mask] ** (1 / 2.4)) - 0.055
    return out

def srgb_to_lin(img):
    img = img.astype(np.float32) / 255.0
    mask = img <= 0.04045
    out = np.empty_like(img)
    out[mask] = img[mask] / 12.92
    out[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return np.clip(out, 0, 1)
def plot_psf_visuals(psf_res, factor, out_dir):
    if not psf_res.get("accepted_edges"): return
    
    # Draw simple fit plot with cv2
    best_edge = min(psf_res['accepted_edges'], key=lambda e: e['rmse'])
    
    h, w = 300, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    t = np.array(best_edge['t'])
    profile = np.array(best_edge['profile'])
    fit = np.array(best_edge['fit'])
    
    # Normalize to plot coordinates
    def to_coords(tx, ty):
        tx_norm = (tx - t.min()) / (t.max() - t.min() + 1e-6)
        ty_norm = (ty - profile.min()) / (profile.max() - profile.min() + 1e-6)
        return int(tx_norm * (w - 20) + 10), int((1 - ty_norm) * (h - 20) + 10)
        
    for i in range(len(t)):
        x, y = to_coords(t[i], profile[i])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        
    for i in range(len(t)-1):
        x1, y1 = to_coords(t[i], fit[i])
        x2, y2 = to_coords(t[i+1], fit[i+1])
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    cv2.putText(img, f"Best PSF Fit {factor}x rmse={best_edge['rmse']:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    Image.fromarray(img).save(os.path.join(out_dir, f"psf_fit_{factor}x.png"))
    
    # Just save kernel
    if "kernel" in psf_res:
        k = psf_res["kernel"]
        k_norm = (k / k.max() * 255).astype(np.uint8)
        k_color = cv2.applyColorMap(k_norm, cv2.COLORMAP_HOT)
        k_resized = cv2.resize(k_color, (200, 200), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(cv2.cvtColor(k_resized, cv2.COLOR_BGR2RGB)).save(os.path.join(out_dir, "psf_kernel.png"))
        
    # Draw simple histogram
    sigmas = [e['sigma'] for e in psf_res['accepted_edges']]
    img_hist = np.ones((h, w, 3), dtype=np.uint8) * 255
    hist, bins = np.histogram(sigmas, bins=10)
    for i in range(10):
        if hist.max() > 0:
            bar_h = int(hist[i] / hist.max() * (h - 40))
            x1 = int(10 + i * (w - 20) / 10)
            x2 = int(10 + (i + 1) * (w - 20) / 10) - 2
            cv2.rectangle(img_hist, (x1, h - 20 - bar_h), (x2, h - 20), (0, 255, 0), -1)
            
    cv2.putText(img_hist, f"Sigma Dist {factor}x", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    Image.fromarray(img_hist).save(os.path.join(out_dir, f"psf_sigma_dist_{factor}x.png"))

def create_contact_sheet(images, labels, cols=2):
    h, w = images[0].shape[:2]
    n = len(images)
    rows = (n + cols - 1) // cols
    
    sheet_h = rows * h + rows * 40
    sheet_w = cols * w
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        y = r * (h + 40)
        x = c * w
        
        sheet[y+40:y+40+h, x:x+w] = img
        cv2.putText(sheet, label, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    return sheet

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
        
    gt_img = np.array(Image.open("m4d_test_assets/porche.png").convert("RGB"))
    gt_h, gt_w = gt_img.shape[:2] # 220, 360
    
    artifacts = []
    def save_artifact(name, img, is_diagnostic=False):
        path = os.path.join(out_dir, name)
        if is_diagnostic and img.ndim == 3:
            img = img.copy()
            cv2.putText(img, "NON-ACCEPTED DIAGNOSTIC OUTPUT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        Image.fromarray(img).save(path)
        artifacts.append({
            "path": f"m5d_results/{name}",
            "size": os.path.getsize(path),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()
        })
        
    def add_existing_artifact(name):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            artifacts.append({
                "path": f"m5d_results/{name}",
                "size": os.path.getsize(path),
                "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()
            })
            
    bench_results = []
    psf_results = {}
    
    regions = {
        "wheel": (100, 205, 185, 260),
        "headlamp": (70, 125, 135, 195),
        "plate": (125, 165, 25, 100)
    }
    
    config_rejection_reasons = {c: [] for c in configs.keys()}
    metrics_by_scale = {}
    
    # 1. Controlled benchmark
    for factor in [2, 4, 8]:
        lw, lh = gt_w // factor, gt_h // factor
        gt_lin = srgb_to_lin(gt_img)
        lr_lin = cv2.resize(gt_lin, (lw, lh), interpolation=cv2.INTER_AREA)
        lr_srgb = (lin_to_srgb(lr_lin) * 255).astype(np.uint8)
        save_artifact(f"lr_controlled_{factor}x.png", lr_srgb)
        
        update_ram()
        
        psf_res = estimate_psf(lr_lin)
        psf_results[f"{factor}x"] = {
            "confidence": psf_res["confidence"],
            "sigma": psf_res["sigma"],
            "reason": psf_res["reason"],
            "accepted_count": psf_res.get("accepted_count", 0)
        }
        
        plot_psf_visuals(psf_res, factor, out_dir)
        add_existing_artifact(f"psf_sigma_dist_{factor}x.png")
        add_existing_artifact(f"psf_fit_{factor}x.png")
        if factor == 2: add_existing_artifact("psf_kernel.png")
        
        overlay = lr_srgb.copy()
        if "accepted_edges" in psf_res:
            for e in psf_res["accepted_edges"]: cv2.circle(overlay, (e['x'], e['y']), 1, (0, 255, 0), -1)
        save_artifact(f"psf_edge_overlay_{factor}x.png", overlay)
            
        update_ram()
        
        base_cand = cv2.resize(lr_lin, (gt_w, gt_h), interpolation=cv2.INTER_LANCZOS4)
        base_srgb = (lin_to_srgb(base_cand) * 255).astype(np.uint8)
        
        metrics_dict = {}
        def calc_metrics(pred_srgb, pred_lin, name):
            m = {}
            m["psnr"] = compute_psnr(gt_img, pred_srgb)
            m["ssim"] = compute_ssim(gt_img, pred_srgb)
            f1 = edge_f1_score(gt_img, pred_srgb)
            m["edge_f1"] = f1["f1"]
            m["ringing_percent"] = calculate_ringing_percentage(gt_img, pred_srgb)
            
            clip_mask = (pred_lin <= 0.0) | (pred_lin >= 1.0)
            m["clipping_percent"] = float(np.mean(clip_mask) * 100.0)
            
            simulated = forward_model(pred_lin, lr_lin.shape, factor, psf_res["sigma"])
            m["source_rmse"] = float(np.sqrt(np.mean((lr_lin - simulated)**2)))
            
            for rname, (y0, y1, x0, x1) in regions.items():
                rm = {}
                gt_r = gt_img[y0:y1, x0:x1]
                pred_r = pred_srgb[y0:y1, x0:x1]
                rm["ssim"] = compute_ssim(gt_r, pred_r)
                m[f"region_{rname}"] = rm
                
            metrics_dict[name] = m
            
        calc_metrics(base_srgb, base_cand, "Lanczos")
        save_artifact(f"controlled_{factor}x_lanczos.png", base_srgb)
        
        contact_images = [gt_img, base_srgb]
        contact_labels = ["Ground Truth", "Lanczos"]
        
        region_contact_images = {rname: [gt_img[y0:y1, x0:x1], base_srgb[y0:y1, x0:x1]] for rname, (y0, y1, x0, x1) in regions.items()}
        region_contact_labels = {rname: ["Ground Truth", "Lanczos"] for rname in regions.keys()}
        
        for cname, cfg in configs.items():
            if psf_res["confidence"] < 0.1:
                m_err, run_ms = 0.0, 0.0
                out_srgb, out_lin = base_srgb, base_cand
                config_rejection_reasons[cname].append(f"{factor}x: Insufficient PSF confidence")
            else:
                t0 = time.time()
                out_lin, history = run_scpb_optimization(lr_lin, base_cand, factor, psf_res["sigma"], cfg, cfg["max_iters"], cfg["initial_step"])
                out_tiled_lin, _, _, _ = reconstruct_tiled(lr_lin, factor, psf_res["sigma"], cfg, cfg["max_iters"], cfg["initial_step"], target_shape=(gt_h, gt_w), tile_size=64)
                run_ms = (time.time() - t0) * 1000
                
                out_srgb = (lin_to_srgb(out_lin) * 255).astype(np.uint8)
                out_tiled_srgb = (lin_to_srgb(out_tiled_lin) * 255).astype(np.uint8)
                m_err, _ = compute_tiling_equivalence_error(out_srgb, out_tiled_srgb)
                
                # Plot objective history
                h_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
                iters = [h['iteration'] for h in history]
                objs = [h['objective'] for h in history]
                if len(iters) > 1:
                    max_obj, min_obj = max(objs), min(objs)
                    for i in range(len(iters)-1):
                        x1 = int(10 + iters[i] / (max(iters)+1e-6) * (400 - 20))
                        y1 = int((1 - (objs[i] - min_obj) / (max_obj - min_obj + 1e-6)) * (300 - 20) + 10)
                        x2 = int(10 + iters[i+1] / (max(iters)+1e-6) * (400 - 20))
                        y2 = int((1 - (objs[i+1] - min_obj) / (max_obj - min_obj + 1e-6)) * (300 - 20) + 10)
                        cv2.line(h_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                cv2.putText(h_img, f"Obj History {factor}x {cname}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                Image.fromarray(h_img).save(os.path.join(out_dir, f"objective_history_{factor}x_{cname}.png"))
                add_existing_artifact(f"objective_history_{factor}x_{cname}.png")
                
                # Residual visual
                simulated = forward_model(out_lin, lr_lin.shape, factor, psf_res["sigma"])
                residual = lr_lin - simulated
                res_vis = np.clip((residual + 0.5) * 255, 0, 255).astype(np.uint8)
                save_artifact(f"residual_vis_{factor}x_{cname}.png", res_vis)
                
            calc_metrics(out_srgb, out_lin, cname)
            metrics_dict[cname]["runtime_ms"] = run_ms
            metrics_dict[cname]["tiling_equivalence_max_error"] = m_err
            
            save_artifact(f"controlled_{factor}x_{cname}_diagnostic.png", out_srgb, is_diagnostic=True)
            
            contact_images.append(out_srgb)
            contact_labels.append(cname)
            for rname, (y0, y1, x0, x1) in regions.items():
                region_contact_images[rname].append(out_srgb[y0:y1, x0:x1])
                region_contact_labels[rname].append(cname)
            
            update_ram()
            
        # Save contact sheets
        sheet = create_contact_sheet(contact_images, contact_labels)
        save_artifact(f"contact_sheet_{factor}x.png", sheet)
        
        for rname in regions.keys():
            sheet_r = create_contact_sheet(region_contact_images[rname], region_contact_labels[rname])
            save_artifact(f"contact_sheet_{rname}_{factor}x.png", sheet_r)
            
        metrics_by_scale[factor] = metrics_dict
        bench_results.append({"scale": factor, "metrics": metrics_dict})
        
    # Check acceptance
    status = "NO_MEASURABLE_GAIN"
    selected_config = "none"
    for cname in configs.keys():
        passed = True
        f1_improvements = 0
        
        for factor in [2, 4, 8]:
            m = metrics_by_scale[factor][cname]
            l_m = metrics_by_scale[factor]["Lanczos"]
            
            if l_m["psnr"] - m["psnr"] > 0.10:
                config_rejection_reasons[cname].append(f"{factor}x: PSNR loss > 0.10 dB")
                passed = False
            if l_m["ssim"] - m["ssim"] > 0.002:
                config_rejection_reasons[cname].append(f"{factor}x: SSIM loss > 0.002")
                passed = False
            if m["edge_f1"] > l_m["edge_f1"]:
                f1_improvements += 1
            if m["ringing_percent"] > l_m["ringing_percent"] + 0.20:
                config_rejection_reasons[cname].append(f"{factor}x: Ringing increased by > 0.20%")
                passed = False
            if m["source_rmse"] > l_m["source_rmse"] + 1e-4:
                config_rejection_reasons[cname].append(f"{factor}x: Source RMSE worse than Lanczos")
                passed = False
            if m["clipping_percent"] >= 0.10:
                config_rejection_reasons[cname].append(f"{factor}x: Clipping >= 0.10%")
                passed = False
            if m.get("tiling_equivalence_max_error", 100) > 1.0:
                config_rejection_reasons[cname].append(f"{factor}x: Tiling error > 1")
                passed = False
                
            for rname in regions.keys():
                if l_m[f"region_{rname}"]["ssim"] - m[f"region_{rname}"]["ssim"] > 0.01:
                    config_rejection_reasons[cname].append(f"{factor}x: Regional SSIM loss > 0.01 in {rname}")
                    passed = False
                    
        if f1_improvements < 2:
            config_rejection_reasons[cname].append("Edge F1 improved in fewer than 2 scales")
            passed = False
            
        if passed and status == "NO_MEASURABLE_GAIN":
            status = "NEEDS_USER_VISUAL_CHECK"
            selected_config = cname
            
    # 2. Operational benchmark (direct from original 360x220)
    original_lin = srgb_to_lin(gt_img)
    psf_res_op = estimate_psf(original_lin)
    
    op_results = {}
    for factor in [2, 4, 8]:
        target_h, target_w = gt_h * factor, gt_w * factor
        op_base_cand = cv2.resize(original_lin, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        op_base_srgb = (lin_to_srgb(op_base_cand) * 255).astype(np.uint8)
        save_artifact(f"operational_{factor}x_lanczos.png", op_base_srgb)
        
        op_res_dict = {}
        for cname, cfg in configs.items():
            if psf_res_op["confidence"] < 0.1:
                op_out_srgb = op_base_srgb
                tile_count = 0
            else:
                op_out_lin, tile_count, halo, p_mem = reconstruct_tiled(
                    original_lin, factor, psf_res_op["sigma"], cfg, cfg["max_iters"], cfg["initial_step"], 
                    target_shape=(target_h, target_w), tile_size=1024
                )
                op_out_srgb = (lin_to_srgb(op_out_lin) * 255).astype(np.uint8)
                if p_mem > peak_ram: peak_ram = p_mem
            
            is_diag = (status != "NEEDS_USER_VISUAL_CHECK" or selected_config != cname)
            save_artifact(f"operational_{factor}x_{cname}_diagnostic.png", op_out_srgb, is_diagnostic=is_diag)
            
            op_res_dict[cname] = {
                "tile_count": tile_count,
                "dimensions": f"{target_w}x{target_h}"
            }
        op_results[f"{factor}x"] = op_res_dict
        update_ram()
        
    results_obj = {
        "status": status,
        "selected_config": selected_config,
        "rejection_reasons": config_rejection_reasons,
        "reconstruction_benchmark": bench_results,
        "operational_results": op_results,
        "psf_results": psf_results,
        "peak_ram_mb": peak_ram / (1024*1024),
        "peak_ram_bytes": peak_ram
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_obj, f, indent=2)
    add_existing_artifact("results.json")
    
    prov_obj = {"run_uuid": run_id, "git_commit": commit}
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov_obj, f, indent=2)
    add_existing_artifact("provenance.json")
    
    ev_obj = {
        "run_uuid": run_id,
        "utc_timestamp": utc_time,
        "git_commit": commit,
        "git_dirty_at_start": dirty,
        "peak_ram_bytes": peak_ram,
        "peak_ram_mb": peak_ram / (1024*1024),
        "status": status,
        "artifacts": artifacts
    }
    
    with open(os.path.join(out_dir, "evidence.jsonl"), "w") as f:
        f.write(json.dumps(ev_obj) + "\n")
        
if __name__ == "__main__":
    main()
