import os
import sys
import json
import time
import uuid
import datetime
import hashlib
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))
from m4d.color import srgb_to_linear, linear_to_srgb, get_luminance
from m4d.metrics import (
    compute_psnr, compute_ssim, edge_f1_score, compute_gradient_energy,
    compute_laplacian_variance, calculate_ringing_percentage
)

from m5_vrc.vector_edges import extract_vector_edges
from m5_vrc.color_field import extract_color_field
from m5_vrc.residual_pyramid import extract_source_residuals
from m5_vrc.scale_renderer import render_scale, enforce_scale_consistency

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

def process_vrc(source_rgb, target_w, target_h, config, return_time=False):
    start = time.time()
    
    # Logical base
    oh, ow = source_rgb.shape[:2]
    source_lin = srgb_to_linear(source_rgb)
    luma = get_luminance(source_lin)
    
    # 1. Vector skeleton
    splines, edge_mask = extract_vector_edges(luma, config['coherence'], config['strength'])
    
    # 2. Color field
    c_field = extract_color_field(source_lin, d=config['d'], sigma_color=config['sig_c'], sigma_space=config['sig_s'])
    
    # 3. Residual
    base_1x = render_scale(c_field, splines, np.zeros_like(source_lin), (ow, oh), (oh, ow), source_lin)
    res_1x = extract_source_residuals(source_lin, base_1x)
    
    # 4. Render
    raw_render = render_scale(c_field, splines, res_1x, (target_w, target_h), (oh, ow), source_lin)
    
    # 5. Consistency
    final_lin = enforce_scale_consistency(raw_render, source_lin, max_iters=5, tolerance=1.0/255.0)
    
    final_srgb = linear_to_srgb(final_lin, to_uint8=True)
    
    if return_time:
        return final_srgb, splines, (time.time() - start) * 1000
    return final_srgb

def main():
    out_dir = "m5_results"
    os.makedirs(out_dir, exist_ok=True)
    
    gt_path = os.path.join("m4d_test_assets", "porche.png")
    gt_img = Image.open(gt_path).convert("RGB")
    gt_arr = np.array(gt_img)
    w, h = gt_img.size # 360, 220
    
    crops_def = {
        "wheel": (185, 100, 260, 205),
        "headlamp": (135, 70, 195, 125)
    }
    
    # Annotated source
    ann = gt_img.copy()
    draw = ImageDraw.Draw(ann)
    for nm, (cx1, cy1, cx2, cy2) in crops_def.items():
        draw.rectangle([cx1, cy1, cx2, cy2], outline="red", width=2)
        draw.text((cx1, cy1 - 10), nm, fill="red")
    ann.save(os.path.join(out_dir, "annotated_source.png"))
    
    # Configs
    configs = {
        "VRC_A": {"coherence": 0.3, "strength": 0.05, "d": 3, "sig_c": 0.05, "sig_s": 1.0},
        "VRC_B": {"coherence": 0.5, "strength": 0.10, "d": 5, "sig_c": 0.10, "sig_s": 2.0},
        "VRC_C": {"coherence": 0.7, "strength": 0.20, "d": 7, "sig_c": 0.20, "sig_s": 3.0}
    }
    
    # Part A: Ground truth benchmark
    scales = [{"n": "2x", "w": 180, "h": 110}, {"n": "4x", "w": 90, "h": 55}, {"n": "8x", "w": 45, "h": 28}]
    bench_res = []
    
    best_config = "VRC_B"
    
    for sc in scales:
        s_n, s_w, s_h = sc["n"], sc["w"], sc["h"]
        lr = cv2.resize(gt_arr, (s_w, s_h), interpolation=cv2.INTER_AREA)
        
        metrics_dict = {}
        
        # Lanczos
        st = time.time()
        lz = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        rt = (time.time() - st) * 1000
        
        ef1 = edge_f1_score(gt_arr, lz)
        metrics_dict["Lanczos"] = {
            "psnr": compute_psnr(gt_arr, lz),
            "ssim": compute_ssim(gt_arr, lz),
            "edge_f1": ef1["f1"],
            "ringing_percent": calculate_ringing_percentage(gt_arr, lz),
            "runtime_ms": rt
        }
        
        for c_name, cfg in configs.items():
            vrc, spl, rt = process_vrc(lr, w, h, cfg, return_time=True)
            
            # Save overlays for VRC_B
            if c_name == best_config and s_n == "2x":
                # Save skeleton visualization on base
                skel_img = vrc.copy()
                from m5_vrc.scale_renderer import evaluate_splines_on_grid
                m = evaluate_splines_on_grid(spl, (w, h), (s_h, s_w))
                skel_img[m > 0.1] = [0, 255, 0]
                Image.fromarray(skel_img).save(os.path.join(out_dir, "fitted_curve_overlay.png"))
                
            ef1 = edge_f1_score(gt_arr, vrc)
            metrics_dict[c_name] = {
                "psnr": compute_psnr(gt_arr, vrc),
                "ssim": compute_ssim(gt_arr, vrc),
                "edge_f1": ef1["f1"],
                "ringing_percent": calculate_ringing_percentage(gt_arr, vrc),
                "runtime_ms": rt
            }
            
        bench_res.append({"scale": s_n, "metrics": metrics_dict})
        
    # Part B: Operational Zoom
    enlarge_scales = [2, 4, 8]
    op_metrics = []
    
    for factor in enlarge_scales:
        ew, eh = w * factor, h * factor
        
        # Lanczos
        lz = cv2.resize(gt_arr, (ew, eh), interpolation=cv2.INTER_LANCZOS4)
        Image.fromarray(lz).save(os.path.join(out_dir, f"enlarged_{factor}x_Lanczos.png"))
        
        # VRC Best
        vrc = process_vrc(gt_arr, ew, eh, configs[best_config])
        Image.fromarray(vrc).save(os.path.join(out_dir, f"enlarged_{factor}x_VRC.png"))
        
        # Calculate consistency error against ground truth (the parent)
        vrc_down = cv2.resize(vrc, (w, h), interpolation=cv2.INTER_AREA)
        err = np.abs(gt_arr.astype(np.float32) - vrc_down.astype(np.float32))
        max_err = float(err.max())
        mean_err = float(err.mean())
        
        op_metrics.append({
            "scale": f"{factor}x",
            "max_consistency_error": max_err,
            "mean_consistency_error": mean_err
        })
        
        # Crops
        for c_nm, (cx1, cy1, cx2, cy2) in crops_def.items():
            for m_nm, arr in [("Lanczos", lz), ("VRC", vrc)]:
                cr = arr[cy1*factor:cy2*factor, cx1*factor:cx2*factor]
                Image.fromarray(cr).save(os.path.join(out_dir, f"crop_{factor}x_{c_nm}_{m_nm}.png"))

    # Contact sheets
    for c_nm in crops_def:
        for factor in enlarge_scales:
            imgs = {
                "Lanczos": Image.open(os.path.join(out_dir, f"crop_{factor}x_{c_nm}_Lanczos.png")),
                "VRC": Image.open(os.path.join(out_dir, f"crop_{factor}x_{c_nm}_VRC.png"))
            }
            make_contact_sheet(imgs, f"{c_nm} {factor}x", os.path.join(out_dir, f"contact_{factor}x_{c_nm}.png"))
            
    # Evaluation Logic
    m2x = bench_res[0]["metrics"]
    vrc_m = m2x[best_config]
    lz_m = m2x["Lanczos"]
    
    f1_win = vrc_m["edge_f1"] >= lz_m["edge_f1"]
    psnr_loss = lz_m["psnr"] - vrc_m["psnr"]
    cons_ok = all(x["max_consistency_error"] <= 1.5 for x in op_metrics) # integer rounding 
    
    if f1_win and psnr_loss <= 0.25 and cons_ok:
        status = "NEEDS_USER_VISUAL_CHECK"
    else:
        status = "NO_MEASURABLE_GAIN"
        
    results_obj = {
        "status": status,
        "best_config": best_config,
        "reconstruction_benchmark": bench_res,
        "operational_metrics": op_metrics
    }
    
    with open(os.path.join(out_dir, "results.json"), "w", newline='\n') as f:
        json.dump(results_obj, f, indent=2)
        f.write('\n')
        # Generate Experimental Package
    import zipfile
    import pickle
    pkg_path = os.path.join(out_dir, "vrc_package.zip")
    with zipfile.ZipFile(pkg_path, 'w') as zf:
        # Source image
        zf.write(gt_path, arcname="source.png")
        # Metadata
        meta = {
            "logical_dimensions": [w, h],
            "supported_scales": ["1x", "2x", "4x", "8x"],
            "color_field_params": configs[best_config],
            "maximum_reliable_scale_per_region": {"global": "4x", "edges": "8x"},
            "algorithm_version": "vrc-math-poc-v1",
            "source_hash": hash_file(gt_path)
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2))
        
        # We need splines for the 1x base
        best_cfg = configs[best_config]
        gt_lin = srgb_to_linear(gt_arr)
        gt_luma = get_luminance(gt_lin)
        spl, mask = extract_vector_edges(gt_luma, best_cfg['coherence'], best_cfg['strength'])
        
        # Save splines
        with zf.open("vector_curves.pkl", "w") as f:
            pickle.dump(spl, f)
            
        # Base residual
        c_field = extract_color_field(gt_lin, d=best_cfg['d'], sig_c=best_cfg['sig_c'], sig_s=best_cfg['sig_s'])
        base = render_scale(c_field, spl, np.zeros_like(gt_lin), (w, h), (h, w), gt_lin)
        res = extract_source_residuals(gt_lin, base)
        
        # Tile index
        from m5_vrc.quadtree import generate_tiles
        res_srgb = linear_to_srgb(res, to_uint8=True)
        tiles = generate_tiles(res_srgb)
        
        tile_index = []
        for (tx, ty), t_img in tiles.items():
            t_name = f"tiles/res_0_{tx}_{ty}.png"
            tile_index.append({"z": 0, "x": tx, "y": ty, "path": t_name})
            img_bgr = cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".png", img_bgr)
            zf.writestr(t_name, buf.tobytes())
            
        zf.writestr("quadtree_index.json", json.dumps(tile_index, indent=2))
        
    # Gather artifacts for hashing
    artifacts = [{"path": gt_path, "sha256": hash_file(gt_path)}]
    for fname in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p) and fname != "evidence.jsonl":
            artifacts.append({"path": p, "sha256": hash_file(p)})
            
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
    
    with open(os.path.join(out_dir, "evidence.jsonl"), "w", newline='\n') as f:
        f.write(json.dumps(ev_obj) + "\n")
        
    print(f"Status: {status}")

if __name__ == "__main__":
    main()
