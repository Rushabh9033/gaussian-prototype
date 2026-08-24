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
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))
from m4d.color import srgb_to_linear, linear_to_srgb, get_luminance
from m4d.metrics import (
    compute_psnr, compute_ssim, edge_f1_score, compute_gradient_energy,
    compute_laplacian_variance, calculate_ringing_percentage
)

from m5b_edr.structure_tensor import extract_edge_orientation
from m5b_edr.tiled_renderer import render_tiled
from m5b_edr.metrics import color_histogram_distance, compute_tile_seam_error

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
        return commit, dirty
    except Exception:
        return "unknown", False

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def make_contact_sheet(images_dict, title, out_path):
    w, h = list(images_dict.values())[0].size
    n = len(images_dict)
    label_h = 30
    sheet = Image.new("RGB", (w * n, h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(images_dict.items()):
        sheet.paste(img, (i * w, label_h))
        draw.text((i * w + 5, 5), name, fill="black")
    sheet.save(out_path)

def process_edr(source_rgb, target_w, target_h, config):
    start = time.time()
    
    source_lin = srgb_to_linear(source_rgb)
    luma = get_luminance(source_lin)
    
    tensor_data = extract_edge_orientation(luma, sigma=1.0)
    
    out_lin = render_tiled(source_lin, target_w, target_h, 
                           tensor_data["theta"], tensor_data["coherence"], tensor_data["uncertainty"],
                           config, tile_size=256)
    
    out_lin = np.clip(out_lin, 0.0, 1.0)
    out_srgb = linear_to_srgb(out_lin, to_uint8=True)
    
    rt = (time.time() - start) * 1000
    
    # Save intermediate visualisations only for the 1x scale if requested, but for now we just return the data
    return out_srgb, out_lin, tensor_data, rt

def compute_region_metrics(gt_arr, pred_arr, prefix=""):
    ef1 = edge_f1_score(gt_arr, pred_arr)
    return {
        f"{prefix}psnr": compute_psnr(gt_arr, pred_arr),
        f"{prefix}ssim": compute_ssim(gt_arr, pred_arr),
        f"{prefix}edge_precision": ef1.get("precision", 0.0),
        f"{prefix}edge_recall": ef1.get("recall", 0.0),
        f"{prefix}edge_f1": ef1["f1"],
        f"{prefix}gradient_energy": compute_gradient_energy(pred_arr),
        f"{prefix}laplacian_variance": compute_laplacian_variance(pred_arr),
        f"{prefix}ringing_percent": calculate_ringing_percentage(gt_arr, pred_arr),
        f"{prefix}color_hist_dist": color_histogram_distance(gt_arr, pred_arr)
    }

def main():
    out_dir = "m5b_results"
    os.makedirs(out_dir, exist_ok=True)
    
    gt_path = os.path.join("m4d_test_assets", "porche.png")
    gt_img = Image.open(gt_path).convert("RGB")
    gt_arr = np.array(gt_img)
    w, h = gt_img.size
    
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
    
    proc = psutil.Process()
    peak_ram = proc.memory_info().rss
    
    def update_ram():
        nonlocal peak_ram
        peak_ram = max(peak_ram, proc.memory_info().rss)

    # Visualization of intermediate maps for the 1x image
    _, _, tdata, _ = process_edr(gt_arr, w, h, configs["EDR_B"])
    Image.fromarray((tdata["strength"] * 255).astype(np.uint8)).save(os.path.join(out_dir, "vis_strength.png"))
    Image.fromarray((tdata["coherence"] * 255).astype(np.uint8)).save(os.path.join(out_dir, "vis_coherence.png"))
    Image.fromarray((tdata["uncertainty"] * 255).astype(np.uint8)).save(os.path.join(out_dir, "vis_uncertainty.png"))
    # Normalize theta from [-pi, pi] to [0, 255]
    theta_norm = ((tdata["theta"] + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    Image.fromarray(theta_norm).save(os.path.join(out_dir, "vis_theta.png"))

    for sc in scales:
        s_n, s_w, s_h = sc["n"], sc["w"], sc["h"]
        # Explicit area downsampling (simulated low-res sensor)
        lr = cv2.resize(gt_arr, (s_w, s_h), interpolation=cv2.INTER_AREA)
        
        metrics_dict = {}
        
        # Bicubic
        st = time.time()
        bc = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        bc_rt = (time.time() - st) * 1000
        metrics_dict["Bicubic"] = compute_region_metrics(gt_arr, bc, "global_")
        metrics_dict["Bicubic"]["runtime_ms"] = bc_rt
        Image.fromarray(bc).save(os.path.join(out_dir, f"recon_{s_n}_Bicubic.png"))
        
        # Lanczos
        st = time.time()
        lz = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        lz_rt = (time.time() - st) * 1000
        metrics_dict["Lanczos"] = compute_region_metrics(gt_arr, lz, "global_")
        metrics_dict["Lanczos"]["runtime_ms"] = lz_rt
        Image.fromarray(lz).save(os.path.join(out_dir, f"recon_{s_n}_Lanczos.png"))
        
        # EDR configs
        recons = {"Bicubic": bc, "Lanczos": lz}
        for c_name, cfg in configs.items():
            edr_out, _, _, rt = process_edr(lr, w, h, cfg)
            m_edr = compute_region_metrics(gt_arr, edr_out, "global_")
            m_edr["runtime_ms"] = rt
            
            seam_max, seam_mean = compute_tile_seam_error(edr_out, 256)
            m_edr["tile_seam_max_error"] = seam_max
            m_edr["tile_seam_mean_error"] = seam_mean
            
            metrics_dict[c_name] = m_edr
            recons[c_name] = edr_out
            Image.fromarray(edr_out).save(os.path.join(out_dir, f"recon_{s_n}_{c_name}.png"))
            
            update_ram()
            
        bench_res.append({"scale": s_n, "metrics": metrics_dict})
        global_metrics[s_n] = metrics_dict
        
        # Contact sheets for this scale
        for c_nm, (cx1, cy1, cx2, cy2) in crops_def.items():
            imgs = {"Reference": Image.fromarray(gt_arr[cy1:cy2, cx1:cx2])}
            for m_nm, arr in recons.items():
                imgs[m_nm] = Image.fromarray(arr[cy1:cy2, cx1:cx2])
            make_contact_sheet(imgs, f"{c_nm} {s_n}", os.path.join(out_dir, f"contact_recon_{s_n}_{c_nm}.png"))

    # Config Selection Rules
    # Eligible if PSNR loss <= 0.25 dB at 2x, SSIM loss <= 0.005 at 2x, ringing <= +0.5% at 2x, no seams (>2 error).
    m2x = global_metrics["2x"]
    lz2x = m2x["Lanczos"]
    
    eligible = []
    for c_name in configs:
        c2x = m2x[c_name]
        psnr_loss = lz2x["global_psnr"] - c2x["global_psnr"]
        ssim_loss = lz2x["global_ssim"] - c2x["global_ssim"]
        ring_diff = c2x["global_ringing_percent"] - lz2x["global_ringing_percent"]
        seam_ok = c2x.get("tile_seam_max_error", 0) <= 2.0
        
        if psnr_loss <= 0.25 and ssim_loss <= 0.005 and ring_diff <= 0.5 and seam_ok:
            eligible.append(c_name)
            
    best_config = None
    if eligible:
        best_avg_f1 = -1
        for c_name in eligible:
            avg_f1 = np.mean([global_metrics[s["n"]][c_name]["global_edge_f1"] for s in scales])
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_config = c_name
                
    status = "NEEDS_USER_VISUAL_CHECK" if best_config else "NO_MEASURABLE_GAIN"
    sel_config = best_config if best_config else "none"

    # Operational test
    enlarge_scales = [2, 4, 8]
    if best_config:
        for factor in enlarge_scales:
            ew, eh = w * factor, h * factor
            lz = cv2.resize(gt_arr, (ew, eh), interpolation=cv2.INTER_LANCZOS4)
            Image.fromarray(lz).save(os.path.join(out_dir, f"op_{factor}x_Lanczos.png"))
            
            edr_out, _, _, _ = process_edr(gt_arr, ew, eh, configs[best_config])
            Image.fromarray(edr_out).save(os.path.join(out_dir, f"op_{factor}x_EDR.png"))
            
            update_ram()

    commit, dirty = get_git_info()
    
    results_obj = {
        "status": status,
        "selected_config": sel_config,
        "peak_ram_bytes": peak_ram,
        "reconstruction_benchmark": bench_res
    }
    
    artifacts = [{"path": gt_path, "sha256": hash_file(gt_path)}]
    for fname in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p) and fname not in ["evidence.jsonl", "results.json"]:
            artifacts.append({"path": p, "sha256": hash_file(p)})
            
    ev_obj = {
        "run_uuid": str(uuid.uuid4()),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "command_arguments": sys.argv,
        "source_sha256": artifacts[0]["sha256"],
        "algorithm_version": "edr-v1",
        "configs": configs,
        "dependencies": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__
        },
        "hardware": f"{platform.system()} {platform.machine()}",
        "available_ram_bytes": psutil.virtual_memory().total,
        "peak_ram_bytes": peak_ram,
        "selected_config": sel_config,
        "results": results_obj,
        "artifacts": artifacts
    }
    
    with open(os.path.join(out_dir, "results.json"), "w", newline='\n') as f:
        json.dump(results_obj, f, indent=2)
        f.write('\n')
        
    with open(os.path.join(out_dir, "evidence.jsonl"), "w", newline='\n') as f:
        f.write(json.dumps(ev_obj) + "\n")

    print(f"Status: {status}")

if __name__ == "__main__":
    main()
