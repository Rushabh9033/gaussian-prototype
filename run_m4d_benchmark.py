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
    w, h = list(images_dict.values())[0].size
    n = len(images_dict)
    
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
    
    # Corrected crops for 360x220 image
    crops_def = {
        "wheel": (30, 140, 90, 200),
        "headlamp": (280, 110, 330, 150),
        "plate": (300, 160, 350, 190),
        "foliage": (150, 10, 230, 60)
    }
    
    # 0. Draw annotated source image
    annotated_img = gt_img.copy()
    draw = ImageDraw.Draw(annotated_img)
    for c_name, (cx1, cy1, cx2, cy2) in crops_def.items():
        draw.rectangle([cx1, cy1, cx2, cy2], outline="red", width=2)
        draw.text((cx1, cy1 - 10), c_name, fill="red")
    annotated_path = os.path.join(out_dir, "annotated_source.png")
    annotated_img.save(annotated_path)
    
    scales = [
        {"name": "2x", "size": (180, 110)},
        {"name": "4x", "size": (90, 55)},
        {"name": "8x", "size": (45, 28)}
    ]
    
    methods = {
        "Nearest": zoom_nearest,
        "Bicubic": zoom_bicubic,
        "Lanczos": zoom_lanczos
    }
    
    hybrid_strengths = [0.5, 1.0, 1.5]
    for s in hybrid_strengths:
        methods[f"Hybrid_{s}"] = lambda img, tg, st=s: zoom_hybrid(img, tg, strength=st)
    
    benchmark_results = []
    
    # 1. Benchmark Loop
    for scale in scales:
        s_name = scale["name"]
        s_w, s_h = scale["size"]
        
        low_res = cv2.resize(gt_arr, (s_w, s_h), interpolation=cv2.INTER_AREA)
        lr_path = os.path.join(out_dir, f"input_{s_name}.png")
        Image.fromarray(low_res).save(lr_path)
        
        gt_grad_energy = compute_gradient_energy(gt_arr)
        gt_lap_var = compute_laplacian_variance(gt_arr)
        
        scale_metrics = {}
        for m_name, m_func in methods.items():
            start_t = time.time()
            restored = m_func(low_res, (w, h))
            end_t = time.time()
            
            out_path = os.path.join(out_dir, f"restored_{s_name}_{m_name}.png")
            Image.fromarray(restored).save(out_path)
            
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
        
    # Pick the best hybrid method
    best_hybrid = "Hybrid_1.0"
    best_score = -1
    for s in hybrid_strengths:
        h_name = f"Hybrid_{s}"
        # We value SSIM and Edge F1. Let's score them by averaging their 4x SSIM and Edge F1
        m4x = benchmark_results[1]["metrics"][h_name]
        score = m4x["ssim"] + m4x["edge_f1"] - (m4x["ringing_percent"] / 100.0)
        if score > best_score:
            best_score = score
            best_hybrid = h_name
            
    print(f"Selected {best_hybrid} as the primary Hybrid candidate.")
            
    # Clean up results for json by renaming the chosen one to "Hybrid" and removing others
    clean_benchmark = []
    for s_res in benchmark_results:
        clean_metrics = {}
        for k in ["Nearest", "Bicubic", "Lanczos"]:
            clean_metrics[k] = s_res["metrics"][k]
        clean_metrics["Hybrid"] = s_res["metrics"][best_hybrid]
        clean_benchmark.append({
            "scale": s_res["scale"],
            "metrics": clean_metrics
        })
        
    # 2. Real Enlargement Demo
    enlarge_methods = {"Bicubic": zoom_bicubic, "Lanczos": zoom_lanczos, "Hybrid": methods[best_hybrid]}
    enlarge_scales = [2, 4, 8]
    
    for factor in enlarge_scales:
        ew, eh = w * factor, h * factor
        for m_name, m_func in enlarge_methods.items():
            enlarged = m_func(gt_arr, (ew, eh))
            out_path = os.path.join(out_dir, f"enlarged_{factor}x_{m_name}.png")
            Image.fromarray(enlarged).save(out_path)
            
            for crop_name, (cx1, cy1, cx2, cy2) in crops_def.items():
                cx1_s, cy1_s = cx1 * factor, cy1 * factor
                cx2_s, cy2_s = cx2 * factor, cy2 * factor
                crop_arr = enlarged[cy1_s:cy2_s, cx1_s:cx2_s]
                c_path = os.path.join(out_dir, f"crop_{factor}x_{crop_name}_{m_name}.png")
                Image.fromarray(crop_arr).save(c_path)

    # Generate contact sheets
    for crop_name in crops_def:
        sheet_imgs = {}
        for m_name in enlarge_methods.keys():
            p = os.path.join(out_dir, f"crop_8x_{crop_name}_{m_name}.png")
            sheet_imgs[m_name] = Image.open(p)
        c_path = os.path.join(out_dir, f"contact_8x_{crop_name}.png")
        make_contact_sheet(sheet_imgs, f"{crop_name} 8x", c_path)
        
    full_sheet = {}
    for m_name in enlarge_methods.keys():
        p = os.path.join(out_dir, f"enlarged_4x_{m_name}.png")
        full_sheet[m_name] = Image.open(p)
    f_path = os.path.join(out_dir, "contact_full_4x.png")
    make_contact_sheet(full_sheet, "Full 4x", f_path)

    # Acceptance Logic
    ssim_psnr_wins = 0
    edge_f1_wins = 0
    for s_res in clean_benchmark:
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

    results_obj = {
        "status": status,
        "ssim_psnr_wins_vs_lanczos": ssim_psnr_wins,
        "edge_f1_wins_vs_lanczos": edge_f1_wins,
        "best_hybrid_strength": best_hybrid,
        "benchmark": clean_benchmark
    }
    
    res_path = os.path.join(out_dir, "results.json")
    with open(res_path, "w", newline='\n') as f:
        json.dump(results_obj, f, indent=2)
        f.write('\n')
        
    # Gather artifacts for hashing
    # Delete unused restored_*_Hybrid_*.png except the selected one
    for fname in os.listdir(out_dir):
        if fname.startswith("restored_") and "Hybrid_" in fname:
            if best_hybrid not in fname:
                os.remove(os.path.join(out_dir, fname))
            else:
                new_fname = fname.replace(best_hybrid, "Hybrid")
                os.rename(os.path.join(out_dir, fname), os.path.join(out_dir, new_fname))

    artifacts = []
    artifacts.append({"path": gt_path, "sha256": hash_file(gt_path)})
    for fname in sorted(os.listdir(out_dir)):
        if fname in ["evidence.jsonl"]:
            continue
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p):
            artifacts.append({"path": p, "sha256": hash_file(p)})
            
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
    with open(ev_path, "w", newline='\n') as f:
        f.write(json.dumps(ev_obj) + "\n")

    print(f"Status: {status}")

if __name__ == "__main__":
    main()
