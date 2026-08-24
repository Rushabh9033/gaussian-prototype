import os
import json
import time
import hashlib
import uuid
import sys
import datetime
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))
from m4d.zoom import zoom_nearest, zoom_bicubic, zoom_lanczos, zoom_hybrid
from m4d.metrics import (
    compute_psnr, compute_ssim, edge_f1_score, compute_gradient_energy,
    compute_laplacian_variance, calculate_ringing_percentage
)

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

def make_contact_sheet(images_dict, title, out_path):
    # images_dict: dict of title -> Image.Image
    w, h = list(images_dict.values())[0].size
    n = len(images_dict)
    
    # Add space for labels
    label_h = 30
    sheet = Image.new("RGB", (w * n, h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    
    for i, (name, img) in enumerate(images_dict.items()):
        sheet.paste(img, (i * w, label_h))
        draw.text((i * w + 5, 5), name, fill="black")
        
    sheet.save(out_path)

def main():
    out_dir = "m4d_results"
    os.makedirs(out_dir, exist_ok=True)
    
    gt_path = os.path.join("m4d_test_assets", "porche.png")
    if not os.path.exists(gt_path):
        print(f"Missing {gt_path}")
        sys.exit(1)
        
    gt_img = Image.open(gt_path).convert("RGB")
    gt_arr = np.array(gt_img)
    w, h = gt_img.size  # 360, 220
    
    scales = [
        {"name": "2x", "size": (180, 110)},
        {"name": "4x", "size": (90, 55)},
        {"name": "8x", "size": (45, 28)}
    ]
    
    methods = {
        "Nearest": zoom_nearest,
        "Bicubic": zoom_bicubic,
        "Lanczos": zoom_lanczos,
        "Hybrid": zoom_hybrid
    }
    
    benchmark_results = []
    artifacts = []
    artifacts.append({"path": gt_path, "sha256": hash_file(gt_path)})
    
    # 1. Benchmark Loop
    for scale in scales:
        s_name = scale["name"]
        s_w, s_h = scale["size"]
        
        # Downsample using area averaging
        low_res = cv2.resize(gt_arr, (s_w, s_h), interpolation=cv2.INTER_AREA)
        lr_path = os.path.join(out_dir, f"input_{s_name}.png")
        Image.fromarray(low_res).save(lr_path)
        artifacts.append({"path": lr_path, "sha256": hash_file(lr_path)})
        
        gt_grad_energy = compute_gradient_energy(gt_arr)
        gt_lap_var = compute_laplacian_variance(gt_arr)
        
        scale_metrics = {}
        for m_name, m_func in methods.items():
            start_t = time.time()
            restored = m_func(low_res, (w, h))
            end_t = time.time()
            
            out_path = os.path.join(out_dir, f"restored_{s_name}_{m_name}.png")
            Image.fromarray(restored).save(out_path)
            artifacts.append({"path": out_path, "sha256": hash_file(out_path)})
            
            edge_stats = edge_f1_score(gt_arr, restored)
            m_grad_energy = compute_gradient_energy(restored)
            m_lap_var = compute_laplacian_variance(restored)
            
            m_res = {
                "psnr": compute_psnr(gt_arr, restored),
                "ssim": compute_ssim(gt_arr, restored),
                "edge_f1": edge_stats["f1"],
                "edge_precision": edge_stats["precision"],
                "edge_recall": edge_stats["recall"],
                "gradient_energy_error": abs(gt_grad_energy - m_grad_energy) / (gt_grad_energy + 1e-9),
                "laplacian_variance_error": abs(gt_lap_var - m_lap_var) / (gt_lap_var + 1e-9),
                "ringing_percent": calculate_ringing_percentage(gt_arr, restored),
                "runtime_ms": (end_t - start_t) * 1000
            }
            scale_metrics[m_name] = m_res
            
        benchmark_results.append({
            "scale": s_name,
            "metrics": scale_metrics
        })
        
    # 2. Real Enlargement Demo
    # Enlarge 360x220 to 2x, 4x, 8x using Bicubic, Lanczos, Hybrid
    enlarge_methods = ["Bicubic", "Lanczos", "Hybrid"]
    enlarge_scales = [2, 4, 8]
    
    # Crops at 360x220 scale
    crops_def = {
        "wheel": (30, 110, 100, 180),
        "headlamp": (30, 70, 70, 110),
        "plate": (5, 160, 45, 190),
        "foliage": (200, 10, 300, 80)
    }
    
    for factor in enlarge_scales:
        ew, eh = w * factor, h * factor
        for m_name in enlarge_methods:
            m_func = methods[m_name]
            enlarged = m_func(gt_arr, (ew, eh))
            out_path = os.path.join(out_dir, f"enlarged_{factor}x_{m_name}.png")
            Image.fromarray(enlarged).save(out_path)
            artifacts.append({"path": out_path, "sha256": hash_file(out_path)})
            
            # Extract crops scaled up by factor
            for crop_name, (cx1, cy1, cx2, cy2) in crops_def.items():
                cx1_s, cy1_s = cx1 * factor, cy1 * factor
                cx2_s, cy2_s = cx2 * factor, cy2 * factor
                crop_arr = enlarged[cy1_s:cy2_s, cx1_s:cx2_s]
                c_path = os.path.join(out_dir, f"crop_{factor}x_{crop_name}_{m_name}.png")
                Image.fromarray(crop_arr).save(c_path)
                artifacts.append({"path": c_path, "sha256": hash_file(c_path)})

    # Generate contact sheets for crops at 8x
    for crop_name in crops_def:
        sheet_imgs = {}
        for m_name in enlarge_methods:
            p = os.path.join(out_dir, f"crop_8x_{crop_name}_{m_name}.png")
            sheet_imgs[m_name] = Image.open(p)
        c_path = os.path.join(out_dir, f"contact_8x_{crop_name}.png")
        make_contact_sheet(sheet_imgs, f"{crop_name} 8x", c_path)
        artifacts.append({"path": c_path, "sha256": hash_file(c_path)})
        
    # Generate full image contact sheet at 4x
    full_sheet = {}
    for m_name in enlarge_methods:
        p = os.path.join(out_dir, f"enlarged_4x_{m_name}.png")
        full_sheet[m_name] = Image.open(p)
    f_path = os.path.join(out_dir, "contact_full_4x.png")
    make_contact_sheet(full_sheet, "Full 4x", f_path)
    artifacts.append({"path": f_path, "sha256": hash_file(f_path)})

    # Acceptance Logic
    # Hybrid beats or matches Lanczos SSIM/PSNR on at least two benchmark scales
    # Improves edge F1 on at least two scales
    ssim_psnr_wins = 0
    edge_f1_wins = 0
    for s_res in benchmark_results:
        m = s_res["metrics"]
        h_ssim = m["Hybrid"]["ssim"]
        l_ssim = m["Lanczos"]["ssim"]
        h_psnr = m["Hybrid"]["psnr"]
        l_psnr = m["Lanczos"]["psnr"]
        
        if h_ssim >= l_ssim or h_psnr >= l_psnr:
            ssim_psnr_wins += 1
            
        if m["Hybrid"]["edge_f1"] > m["Lanczos"]["edge_f1"]:
            edge_f1_wins += 1
            
    if ssim_psnr_wins >= 2 and edge_f1_wins >= 2:
        status = "NEEDS_USER_VISUAL_CHECK"
    else:
        status = "CLASSICAL_LIMIT_CONFIRMED"

    # Save results
    results_obj = {
        "status": status,
        "ssim_psnr_wins_vs_lanczos": ssim_psnr_wins,
        "edge_f1_wins_vs_lanczos": edge_f1_wins,
        "benchmark": benchmark_results
    }
    res_path = os.path.join(out_dir, "results.json")
    with open(res_path, "w") as f:
        json.dump(results_obj, f, indent=2)
    artifacts.append({"path": res_path, "sha256": hash_file(res_path)})
    
    # Save evidence JSONL
    ev_path = os.path.join(out_dir, "evidence.jsonl")
    ev_obj = {
        "run_uuid": str(uuid.uuid4()),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "command_arguments": sys.argv,
        "source_sha256": artifacts[0]["sha256"],
        "hardware": "CPU/RAM (Non-AI)",
        "results": results_obj,
        "artifacts": artifacts
    }
    with open(ev_path, "w") as f:
        f.write(json.dumps(ev_obj) + "\n")

    print(f"Status: {status}")

if __name__ == "__main__":
    main()
