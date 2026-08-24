import os
import sys
import json
import time
import uuid
import datetime
import hashlib
import subprocess
import psutil
import platform
import traceback
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))
from m4d.color import srgb_to_linear, linear_to_srgb, get_luminance
from m4d.metrics import (
    compute_psnr, compute_ssim, edge_f1_score, compute_gradient_energy,
    compute_laplacian_variance, calculate_ringing_percentage
)

from m5b_edr.structure_tensor import extract_edge_orientation
from m5b_edr.tiled_renderer import render_tiled
from m5b_edr.metrics import color_histogram_distance, compute_tiling_equivalence_error, compute_edge_orientation_error

def get_git_dirty():
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]).strip())
    except Exception:
        return True

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def make_contact_sheet(images_dict, title, out_path, mark_diagnostic=False):
    w, h = list(images_dict.values())[0].size
    n = len(images_dict)
    label_h = 30
    sheet = Image.new("RGB", (w * n, h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(images_dict.items()):
        sheet.paste(img, (i * w, label_h))
        draw.text((i * w + 5, 5), name, fill="black")
        if mark_diagnostic and "EDR" in name:
            draw.text((i * w + 5, h + label_h - 15), "NON-ACCEPTED DIAGNOSTIC OUTPUT", fill="red")
    sheet.save(out_path)

def label_diagnostic(img_arr):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "NON-ACCEPTED DIAGNOSTIC OUTPUT", fill="red")
    return np.array(img)

def process_edr(source_rgb, target_w, target_h, config, tile_size=256):
    start = time.time()
    source_lin = srgb_to_linear(source_rgb)
    luma = get_luminance(source_lin)
    tensor_data = extract_edge_orientation(luma, sigma=1.0)
    out_lin = render_tiled(source_lin, target_w, target_h, 
                           tensor_data["theta"], tensor_data["coherence"], tensor_data["uncertainty"],
                           config, tile_size=tile_size)
    out_lin = np.clip(out_lin, 0.0, 1.0)
    out_srgb = linear_to_srgb(out_lin, to_uint8=True)
    rt = (time.time() - start) * 1000
    return out_srgb, out_lin, tensor_data, rt

def extract_orientation_for_array(arr):
    lin = srgb_to_linear(arr)
    luma = get_luminance(lin)
    return extract_edge_orientation(luma, sigma=1.0)

def compute_all_metrics(gt_arr, pred_arr, gt_tdata, pred_tdata):
    ef1 = edge_f1_score(gt_arr, pred_arr)
    t_gt = gt_tdata["theta"]
    t_pr = pred_tdata["theta"]
    s_gt = gt_tdata["strength"]
    err_rad, err_deg = compute_edge_orientation_error(t_gt, t_pr, s_gt, threshold=0.1)
    
    return {
        "psnr": compute_psnr(gt_arr, pred_arr),
        "ssim": compute_ssim(gt_arr, pred_arr),
        "edge_precision": ef1.get("precision", 0.0),
        "edge_recall": ef1.get("recall", 0.0),
        "edge_f1": ef1["f1"],
        "gradient_energy": compute_gradient_energy(pred_arr),
        "laplacian_variance": compute_laplacian_variance(pred_arr),
        "ringing_percent": calculate_ringing_percentage(gt_arr, pred_arr),
        "color_hist_dist": color_histogram_distance(gt_arr, pred_arr),
        "edge_orientation_err_rad": err_rad,
        "edge_orientation_err_deg": err_deg
    }

def run_benchmark():
    dirty_start = get_git_dirty()
    if dirty_start:
        print("Error: Git is dirty at start.")
        sys.exit(1)
        
    start_commit = get_git_commit()
    out_dir = "m5b_results"
    os.makedirs(out_dir, exist_ok=True)
    
    gt_path = os.path.join("m4d_test_assets", "porche.png")
    gt_img = Image.open(gt_path).convert("RGB")
    gt_arr = np.array(gt_img)
    w, h = gt_img.size
    gt_tdata = extract_orientation_for_array(gt_arr)
    
    crops_def = {
        "wheel": (185, 100, 260, 205),
        "headlamp": (135, 70, 195, 125),
        "plate": (25, 125, 100, 165),
        "smooth": (80, 50, 120, 90),
        "foliage": (0, 0, 50, 50)
    }
    
    configs = {
        "EDR_A": {"tangent_scale": 1.5, "normal_scale": 0.8, "coherence_thresh": 0.6, "photometric_sigma": 0.2},
        "EDR_B": {"tangent_scale": 2.0, "normal_scale": 0.6, "coherence_thresh": 0.4, "photometric_sigma": 0.1},
        "EDR_C": {"tangent_scale": 3.0, "normal_scale": 0.4, "coherence_thresh": 0.2, "photometric_sigma": 0.05}
    }
    
    scales = [{"n": "2x", "w": w//2, "h": h//2}, 
              {"n": "4x", "w": w//4, "h": h//4}, 
              {"n": "8x", "w": w//8, "h": h//8}]
              
    bench_res = []
    global_metrics = {}
    config_breakdown = {c: {} for c in configs}
    
    proc = psutil.Process()
    peak_ram = proc.memory_info().rss
    
    def update_ram():
        nonlocal peak_ram
        peak_ram = max(peak_ram, proc.memory_info().rss)

    # Basic method generation loop
    for sc in scales:
        s_n, s_w, s_h = sc["n"], sc["w"], sc["h"]
        lr = cv2.resize(gt_arr, (s_w, s_h), interpolation=cv2.INTER_AREA)
        
        metrics_dict = {}
        
        # Bicubic
        st = time.time()
        bc = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        bc_rt = (time.time() - st) * 1000
        bc_tdata = extract_orientation_for_array(bc)
        bc_m = compute_all_metrics(gt_arr, bc, gt_tdata, bc_tdata)
        bc_m["runtime_ms"] = bc_rt
        
        # Regional
        for r_name, (cx1, cy1, cx2, cy2) in crops_def.items():
             gt_c = gt_arr[cy1:cy2, cx1:cx2]
             bc_c = bc[cy1:cy2, cx1:cx2]
             gt_t_c = {"theta": gt_tdata["theta"][cy1:cy2, cx1:cx2], "strength": gt_tdata["strength"][cy1:cy2, cx1:cx2]}
             bc_t_c = {"theta": bc_tdata["theta"][cy1:cy2, cx1:cx2], "strength": bc_tdata["strength"][cy1:cy2, cx1:cx2]}
             bc_m[f"region_{r_name}"] = compute_all_metrics(gt_c, bc_c, gt_t_c, bc_t_c)
             
        metrics_dict["Bicubic"] = bc_m
        
        # Lanczos
        st = time.time()
        lz = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        lz_rt = (time.time() - st) * 1000
        lz_tdata = extract_orientation_for_array(lz)
        lz_m = compute_all_metrics(gt_arr, lz, gt_tdata, lz_tdata)
        lz_m["runtime_ms"] = lz_rt
        
        for r_name, (cx1, cy1, cx2, cy2) in crops_def.items():
             gt_c = gt_arr[cy1:cy2, cx1:cx2]
             lz_c = lz[cy1:cy2, cx1:cx2]
             gt_t_c = {"theta": gt_tdata["theta"][cy1:cy2, cx1:cx2], "strength": gt_tdata["strength"][cy1:cy2, cx1:cx2]}
             lz_t_c = {"theta": lz_tdata["theta"][cy1:cy2, cx1:cx2], "strength": lz_tdata["strength"][cy1:cy2, cx1:cx2]}
             lz_m[f"region_{r_name}"] = compute_all_metrics(gt_c, lz_c, gt_t_c, lz_t_c)
             
        metrics_dict["Lanczos"] = lz_m
        
        recons = {"Bicubic": bc, "Lanczos": lz}
        
        # EDR configs
        for c_name, cfg in configs.items():
            edr_out, _, _, rt = process_edr(lr, w, h, cfg, tile_size=256)
            
            # Tiling equivalence check (untiled)
            edr_untiled, _, _, _ = process_edr(lr, w, h, cfg, tile_size=9999)
            te_max, te_mean = compute_tiling_equivalence_error(edr_untiled, edr_out)
            
            edr_tdata = extract_orientation_for_array(edr_out)
            edr_m = compute_all_metrics(gt_arr, edr_out, gt_tdata, edr_tdata)
            edr_m["runtime_ms"] = rt
            edr_m["tiling_equivalence_max_error"] = te_max
            edr_m["tiling_equivalence_mean_error"] = te_mean
            
            for r_name, (cx1, cy1, cx2, cy2) in crops_def.items():
                 gt_c = gt_arr[cy1:cy2, cx1:cx2]
                 edr_c = edr_out[cy1:cy2, cx1:cx2]
                 gt_t_c = {"theta": gt_tdata["theta"][cy1:cy2, cx1:cx2], "strength": gt_tdata["strength"][cy1:cy2, cx1:cx2]}
                 edr_t_c = {"theta": edr_tdata["theta"][cy1:cy2, cx1:cx2], "strength": edr_tdata["strength"][cy1:cy2, cx1:cx2]}
                 edr_m[f"region_{r_name}"] = compute_all_metrics(gt_c, edr_c, gt_t_c, edr_t_c)
                 
            metrics_dict[c_name] = edr_m
            recons[c_name] = edr_out
            update_ram()
            
        bench_res.append({"scale": s_n, "metrics": metrics_dict})
        global_metrics[s_n] = metrics_dict

    # Configuration Selection & Breakdown
    m2x = global_metrics["2x"]
    lz2x = m2x["Lanczos"]
    
    eligible = []
    diagnostic_config = "EDR_A"
    best_f1_for_diag = -1
    
    for c_name in configs:
        c2x = m2x[c_name]
        psnr_loss = lz2x["psnr"] - c2x["psnr"]
        ssim_loss = lz2x["ssim"] - c2x["ssim"]
        ring_diff = c2x["ringing_percent"] - lz2x["ringing_percent"]
        te_err = c2x["tiling_equivalence_max_error"]
        
        f1_improvements = 0
        avg_f1 = 0
        for s in scales:
            s_n = s["n"]
            edr_f1 = global_metrics[s_n][c_name]["edge_f1"]
            lz_f1 = global_metrics[s_n]["Lanczos"]["edge_f1"]
            avg_f1 += edr_f1
            if edr_f1 > lz_f1:
                f1_improvements += 1
        avg_f1 /= 3.0
        
        if avg_f1 > best_f1_for_diag:
            best_f1_for_diag = avg_f1
            diagnostic_config = c_name
        
        cond_psnr = (psnr_loss <= 0.25)
        cond_ssim = (ssim_loss <= 0.005)
        cond_ring = (ring_diff <= 0.5)
        cond_seam = (te_err <= 1e-3) # float tolerance
        
        reasons = []
        if not cond_psnr: reasons.append("PSNR loss > 0.25")
        if not cond_ssim: reasons.append("SSIM loss > 0.005")
        if not cond_ring: reasons.append("Ringing > 0.5%")
        if not cond_seam: reasons.append("Tiling equivalence > 0")
        
        is_eligible = (cond_psnr and cond_ssim and cond_ring and cond_seam)
        if is_eligible: eligible.append(c_name)
        
        config_breakdown[c_name] = {
            "2x_psnr_loss": psnr_loss,
            "2x_ssim_loss": ssim_loss,
            "2x_ringing_diff": ring_diff,
            "tiling_equivalence_error": te_err,
            "edge_f1_improvement_count": f1_improvements,
            "conditions": {
                "psnr_ok": cond_psnr,
                "ssim_ok": cond_ssim,
                "ringing_ok": cond_ring,
                "tiling_ok": cond_seam
            },
            "eligible": is_eligible,
            "rejection_reasons": reasons
        }
        
    best_config = None
    if eligible:
        best_avg_f1 = -1
        for c_name in eligible:
            avg_f1 = np.mean([global_metrics[s["n"]][c_name]["edge_f1"] for s in scales])
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_config = c_name
                
    if best_config and config_breakdown[best_config]["edge_f1_improvement_count"] >= 2:
        status = "NEEDS_USER_VISUAL_CHECK"
    else:
        status = "NO_MEASURABLE_GAIN"
        best_config = None
        
    sel_config = best_config if best_config else "none"

    # Operational test with diagnostic_config
    enlarge_scales = [2, 4, 8]
    op_contact_paths = []
    for factor in enlarge_scales:
        ew, eh = w * factor, h * factor
        lz_arr = cv2.resize(gt_arr, (ew, eh), interpolation=cv2.INTER_LANCZOS4)
        Image.fromarray(lz_arr).save(os.path.join(out_dir, f"op_{factor}x_Lanczos.png"))
        
        edr_out, _, _, _ = process_edr(gt_arr, ew, eh, configs[diagnostic_config], tile_size=256)
        edr_labeled = label_diagnostic(edr_out)
        Image.fromarray(edr_labeled).save(os.path.join(out_dir, f"op_{factor}x_EDR_diagnostic.png"))
        
        # Operational contact sheets for important regions (scaled coordinates)
        op_crops = {
            "wheel": (185*factor, 100*factor, 260*factor, 205*factor),
            "headlamp": (135*factor, 70*factor, 195*factor, 125*factor),
            "plate": (25*factor, 125*factor, 100*factor, 165*factor)
        }
        
        imgs_full = {"Lanczos": Image.fromarray(lz_arr), f"EDR ({diagnostic_config})": Image.fromarray(edr_labeled)}
        make_contact_sheet(imgs_full, "Full", os.path.join(out_dir, f"contact_op_{factor}x_full.png"), mark_diagnostic=True)
        op_contact_paths.append(os.path.join(out_dir, f"contact_op_{factor}x_full.png"))
        
        for crop_name, (cx1, cy1, cx2, cy2) in op_crops.items():
            imgs_crop = {
                "Lanczos": Image.fromarray(lz_arr[cy1:cy2, cx1:cx2]),
                f"EDR ({diagnostic_config})": Image.fromarray(edr_labeled[cy1:cy2, cx1:cx2])
            }
            make_contact_sheet(imgs_crop, crop_name, os.path.join(out_dir, f"contact_op_{factor}x_{crop_name}.png"), mark_diagnostic=True)
        
        update_ram()

    results_obj = {
        "status": status,
        "selected_config": sel_config,
        "diagnostic_config": diagnostic_config,
        "config_selection_breakdown": config_breakdown,
        "reconstruction_benchmark": bench_res
    }
    
    with open(os.path.join(out_dir, "results.json"), "w", newline='\n') as f:
        json.dump(results_obj, f, indent=2)
        f.write('\n')
        
    artifacts = [{"path": gt_path, "sha256": hash_file(gt_path), "size": os.path.getsize(gt_path)}]
    for fname in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p) and fname not in ["evidence.jsonl", "evidence_fail.json", "validation.json"]:
            artifacts.append({"path": p, "sha256": hash_file(p), "size": os.path.getsize(p)})
            
    ev_obj = {
        "run_uuid": str(uuid.uuid4()),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": start_commit,
        "git_dirty_at_start": dirty_start,
        "expected_generated_artifacts": [a["path"] for a in artifacts],
        "command_arguments": sys.argv,
        "source_sha256": artifacts[0]["sha256"],
        "algorithm_version": "edr-v1-corrected",
        "configs": configs,
        "dependencies": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__
        },
        "hardware": f"{platform.system()} {platform.machine()}",
        "cpu_info": platform.processor(),
        "available_ram_bytes": psutil.virtual_memory().available,
        "total_ram_bytes": psutil.virtual_memory().total,
        "peak_ram_bytes": peak_ram,
        "results": results_obj,
        "artifacts": artifacts
    }
    
    with open(os.path.join(out_dir, "evidence.jsonl"), "w", newline='\n') as f:
        f.write(json.dumps(ev_obj) + "\n")

    print(f"Status: {status}")

def main():
    try:
        run_benchmark()
    except Exception as e:
        out_dir = "m5b_results"
        os.makedirs(out_dir, exist_ok=True)
        ev_fail = {
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "traceback": traceback.format_exc(),
            "git_commit": get_git_commit(),
            "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command_arguments": sys.argv
        }
        with open(os.path.join(out_dir, "evidence_fail.json"), "w", newline='\n') as f:
            json.dump(ev_fail, f, indent=2)
            f.write('\n')
        print(f"Benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
