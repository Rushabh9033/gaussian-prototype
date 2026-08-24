import os
import sys
import json
import time
import uuid
import datetime
import hashlib
import subprocess
import zipfile
import io
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

from m5_vrc.vector_edges import extract_vector_edges
from m5_vrc.color_field import extract_color_field
from m5_vrc.residual_pyramid import extract_source_residuals, compute_residual_energy
from m5_vrc.scale_renderer import render_scale, enforce_scale_consistency
from m5_vrc.quadtree import generate_tiles_for_image

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

def process_vrc(source_rgb, target_w, target_h, config, return_metrics=False):
    start = time.time()
    
    oh, ow = source_rgb.shape[:2]
    source_lin = srgb_to_linear(source_rgb)
    luma = get_luminance(source_lin)
    
    splines, edge_mask, rej_count = extract_vector_edges(luma, source_lin, config['coherence'], config['strength'])
    avg_fit_err = float(np.mean([c["fitting_error"] for c in splines])) if splines else 0.0
    
    c_field = extract_color_field(source_lin, d=config['d'], sigma_color=config['sig_c'], sigma_space=config['sig_s'])
    
    base_1x, _ = render_scale(c_field, splines, np.zeros_like(source_lin), (ow, oh), (oh, ow))
    res_1x = extract_source_residuals(source_lin, base_1x)
    res_energy = compute_residual_energy(res_1x)
    
    raw_render, vector_cov = render_scale(c_field, splines, res_1x, (target_w, target_h), (oh, ow))
    
    # 8-bit scale consistency target
    final_lin = enforce_scale_consistency(raw_render, source_lin, max_iters=30, tolerance=0.5/255.0)
    final_srgb = linear_to_srgb(final_lin, to_uint8=True)
    
    rt = (time.time() - start) * 1000
    if return_metrics:
        metrics = {
            "runtime_ms": rt,
            "curve_fitting_error": avg_fit_err,
            "vector_coverage_percent": vector_cov,
            "residual_energy_level_0": res_energy,
            "rejected_curves": rej_count,
            "splines_data": splines,
            "mask": edge_mask,
            "color_field": linear_to_srgb(c_field, to_uint8=True),
            "res_1x": res_1x
        }
        return final_srgb, metrics
    return final_srgb

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
        f"{prefix}ringing_percent": calculate_ringing_percentage(gt_arr, pred_arr)
    }

def create_package(out_path, source_rgb, metadata, splines, residual_levels):
    # Store securely without pickle
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # source
        img_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", img_bgr)
        zf.writestr("source.png", buf.tobytes())
        
        # metadata + splines (pure json)
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        zf.writestr("vector_curves.json", json.dumps(splines, indent=2))
        
        # residual layers as numpy npz (allow_pickle=False)
        io_buf = io.BytesIO()
        np.savez(io_buf, **residual_levels)
        zf.writestr("residuals.npz", io_buf.getvalue())

def main():
    out_dir = "m5_results"
    os.makedirs(out_dir, exist_ok=True)
    
    gt_path = os.path.join("m4d_test_assets", "porche.png")
    gt_img = Image.open(gt_path).convert("RGB")
    gt_arr = np.array(gt_img)
    w, h = gt_img.size
    
    crops_def = {
        "wheel": (185, 100, 260, 205),
        "headlamp": (135, 70, 195, 125)
    }
    
    configs = {
        "VRC_A": {"coherence": 0.3, "strength": 0.05, "d": 3, "sig_c": 0.05, "sig_s": 1.0},
        "VRC_B": {"coherence": 0.5, "strength": 0.10, "d": 5, "sig_c": 0.10, "sig_s": 2.0},
        "VRC_C": {"coherence": 0.7, "strength": 0.20, "d": 7, "sig_c": 0.20, "sig_s": 3.0}
    }
    
    # 1. Ground Truth Benchmark
    scales = [{"n": "2x", "w": 180, "h": 110}, {"n": "4x", "w": 90, "h": 55}, {"n": "8x", "w": 45, "h": 28}]
    bench_res = []
    
    for sc in scales:
        s_n, s_w, s_h = sc["n"], sc["w"], sc["h"]
        lr = cv2.resize(gt_arr, (s_w, s_h), interpolation=cv2.INTER_AREA)
        
        metrics_dict = {}
        
        st = time.time()
        lz = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        lz_rt = (time.time() - st) * 1000
        
        m_lz = compute_region_metrics(gt_arr, lz, "global_")
        m_lz["runtime_ms"] = lz_rt
        metrics_dict["Lanczos"] = m_lz
        
        Image.fromarray(lz).save(os.path.join(out_dir, f"gt_recon_{s_n}_Lanczos.png"))
        
        for c_name, cfg in configs.items():
            vrc, aux = process_vrc(lr, w, h, cfg, return_metrics=True)
            Image.fromarray(vrc).save(os.path.join(out_dir, f"gt_recon_{s_n}_{c_name}.png"))
            
            m_vrc = compute_region_metrics(gt_arr, vrc, "global_")
            m_vrc["runtime_ms"] = aux["runtime_ms"]
            m_vrc["curve_fitting_error"] = aux["curve_fitting_error"]
            m_vrc["vector_coverage_percent"] = aux["vector_coverage_percent"]
            m_vrc["residual_energy_level_0"] = aux["residual_energy_level_0"]
            metrics_dict[c_name] = m_vrc
            
        bench_res.append({"scale": s_n, "metrics": metrics_dict})
        
    # Auto-select configuration based on 2x F1 vs PSNR trade-off
    m2x = bench_res[0]["metrics"]
    lz2x = m2x["Lanczos"]
    
    best_config = None
    best_f1 = -1
    for c_name in configs:
        psnr_loss = lz2x["global_psnr"] - m2x[c_name]["global_psnr"]
        if psnr_loss <= 1.5:  # Tolerance for math POC
            if m2x[c_name]["global_edge_f1"] > best_f1:
                best_f1 = m2x[c_name]["global_edge_f1"]
                best_config = c_name
                
    if best_config is None:
        best_config = "VRC_B"
        
    # 2. Operational Zoom (Scale Consistency) & Visual Evidence
    enlarge_scales = [2, 4, 8]
    op_metrics = []
    
    vrc_1x, aux_1x = process_vrc(gt_arr, w, h, configs[best_config], return_metrics=True)
    Image.fromarray(vrc_1x).save(os.path.join(out_dir, "reconstructed_1x_package.png"))
    Image.fromarray(aux_1x["mask"]).save(os.path.join(out_dir, "detected_edge_mask.png"))
    Image.fromarray(aux_1x["color_field"]).save(os.path.join(out_dir, "color_field_reconstruction.png"))
    
    for factor in enlarge_scales:
        ew, eh = w * factor, h * factor
        lz = cv2.resize(gt_arr, (ew, eh), interpolation=cv2.INTER_LANCZOS4)
        Image.fromarray(lz).save(os.path.join(out_dir, f"enlarged_{factor}x_Lanczos.png"))
        
        vrc, aux = process_vrc(gt_arr, ew, eh, configs[best_config], return_metrics=True)
        Image.fromarray(vrc).save(os.path.join(out_dir, f"enlarged_{factor}x_VRC.png"))
        
        # Consistency error calculated directly on 8-bit output
        vrc_down = cv2.resize(vrc, (w, h), interpolation=cv2.INTER_AREA)
        err = np.abs(gt_arr.astype(np.int16) - vrc_down.astype(np.int16))
        
        op_metrics.append({
            "scale": f"{factor}x",
            "max_consistency_error": float(err.max()),
            "mean_consistency_error": float(err.mean())
        })
        
        # Crops
        for c_nm, (cx1, cy1, cx2, cy2) in crops_def.items():
            for m_nm, arr in [("Lanczos", lz), ("VRC", vrc)]:
                cr = arr[cy1*factor:cy2*factor, cx1*factor:cx2*factor]
                Image.fromarray(cr).save(os.path.join(out_dir, f"crop_{factor}x_{c_nm}_{m_nm}.png"))

    for c_nm in crops_def:
        for factor in enlarge_scales:
            imgs = {
                "Lanczos": Image.open(os.path.join(out_dir, f"crop_{factor}x_{c_nm}_Lanczos.png")),
                "VRC": Image.open(os.path.join(out_dir, f"crop_{factor}x_{c_nm}_VRC.png"))
            }
            make_contact_sheet(imgs, f"{c_nm} {factor}x", os.path.join(out_dir, f"contact_{factor}x_{c_nm}.png"))

    # Generate VRC Package
    pkg_path = os.path.join(out_dir, "vrc_package.zip")
    meta = {
        "logical_dimensions": [w, h],
        "supported_scales": ["1x", "2x"],
        "color_field_params": configs[best_config],
        "maximum_reliable_scale_per_region": {"global": "2x", "edges": "2x"},
        "algorithm_version": "vrc-math-poc-v2",
        "source_hash": hash_file(gt_path)
    }
    # Package actual residual data securely
    residuals = {"level_0": aux_1x["res_1x"]}
    create_package(pkg_path, gt_arr, meta, aux_1x["splines_data"], residuals)

    # Output Size RAM
    proc = psutil.Process()
    peak_ram = proc.memory_info().rss
    pkg_size = os.path.getsize(pkg_path)

    # Evaluation
    f1_win = m2x[best_config]["global_edge_f1"] >= m2x["Lanczos"]["global_edge_f1"]
    psnr_loss = m2x["Lanczos"]["global_psnr"] - m2x[best_config]["global_psnr"]
    cons_ok = all(x["max_consistency_error"] <= 1.5 for x in op_metrics)
    
    status = "NEEDS_USER_VISUAL_CHECK" if (f1_win and psnr_loss <= 0.25 and cons_ok) else "NO_MEASURABLE_GAIN"

    results_obj = {
        "status": status,
        "best_config": best_config,
        "peak_ram_bytes": peak_ram,
        "package_size_bytes": pkg_size,
        "reconstruction_benchmark": bench_res,
        "operational_metrics": op_metrics
    }
    
    with open(os.path.join(out_dir, "results.json"), "w", newline='\n') as f:
        json.dump(results_obj, f, indent=2)
        f.write('\n')

    # Hash Evidence
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
        "configs": configs,
        "dependencies": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "scipy": "installed"
        },
        "hardware": f"{platform.system()} {platform.machine()}",
        "peak_ram": peak_ram,
        "results": results_obj,
        "artifacts": artifacts
    }
    
    with open(os.path.join(out_dir, "evidence.jsonl"), "w", newline='\n') as f:
        f.write(json.dumps(ev_obj) + "\n")

    print(f"Status: {status}")

if __name__ == "__main__":
    main()
