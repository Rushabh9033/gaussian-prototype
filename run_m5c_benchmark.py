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
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))
from m4d.color import srgb_to_linear, linear_to_srgb
from m4d.metrics import (
    compute_psnr, compute_ssim, edge_f1_score, compute_gradient_energy,
    compute_laplacian_variance, calculate_ringing_percentage
)
from m5b_edr.metrics import color_histogram_distance, compute_tiling_equivalence_error

from m5c_isepr import create_scale_space, create_low_surrogate, PatchDictionary, reconstruct_overlap_add

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
    if not os.path.exists(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def make_contact_sheet(images_dict, title, out_path, mark_diagnostic=False, diagnostic_key=None):
    w, h = list(images_dict.values())[0].size
    n = len(images_dict)
    label_h = 30
    sheet = Image.new("RGB", (w * n, h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(images_dict.items()):
        sheet.paste(img, (i * w, label_h))
        draw.text((i * w + 5, 5), name, fill="black")
        if mark_diagnostic and diagnostic_key and diagnostic_key in name:
            draw.text((i * w + 5, h + label_h - 15), "NON-ACCEPTED DIAGNOSTIC OUTPUT", fill="red")
    sheet.save(out_path)

def label_diagnostic(img_arr):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "NON-ACCEPTED DIAGNOSTIC OUTPUT", fill="red")
    return np.array(img)

def compute_all_metrics(gt_arr, pred_arr):
    ef1 = edge_f1_score(gt_arr, pred_arr)
    return {
        "psnr": compute_psnr(gt_arr, pred_arr),
        "ssim": compute_ssim(gt_arr, pred_arr),
        "edge_precision": ef1.get("precision", 0.0),
        "edge_recall": ef1.get("recall", 0.0),
        "edge_f1": ef1["f1"],
        "gradient_energy": compute_gradient_energy(pred_arr),
        "laplacian_variance": compute_laplacian_variance(pred_arr),
        "ringing_percent": calculate_ringing_percentage(gt_arr, pred_arr),
        "color_hist_dist": color_histogram_distance(gt_arr, pred_arr)
    }

def run_benchmark():
    dirty_start = get_git_dirty()
    if dirty_start:
        print("Error: Git is dirty at start.")
        sys.exit(1)
        
    start_commit = get_git_commit()
    out_dir = "m5c_results"
    os.makedirs(out_dir, exist_ok=True)
    
    gt_path = os.path.join("m4d_test_assets", "porche.png")
    gt_img = Image.open(gt_path).convert("RGB")
    gt_arr = np.array(gt_img)
    w, h = gt_img.size
    
    gt_lin = srgb_to_linear(gt_arr)
    
    crops_def = {
        "wheel": (185, 100, 260, 205),
        "headlamp": (135, 70, 195, 125),
        "plate": (25, 125, 100, 165),
        "smooth": (80, 50, 120, 90),
        "foliage": (0, 0, 50, 50),
        "road": (150, 180, 250, 215)
    }
    
    configs = {
        "ISEPR_A": {"patch_size": 5, "top_k": 3, "max_distance": 0.5, "ratio_threshold": 0.8, "residual_gain": 0.5, "max_residual_variance": 0.005, "energy_multiplier": 1.0},
        "ISEPR_B": {"patch_size": 5, "top_k": 5, "max_distance": 1.0, "ratio_threshold": 0.9, "residual_gain": 0.8, "max_residual_variance": 0.01, "energy_multiplier": 1.5},
        "ISEPR_C": {"patch_size": 7, "top_k": 7, "max_distance": 1.5, "ratio_threshold": 0.95, "residual_gain": 1.0, "max_residual_variance": 0.02, "energy_multiplier": 2.0}
    }
    
    scales = [{"n": "2x", "factor": 2}, {"n": "4x", "factor": 4}, {"n": "8x", "factor": 8}]
    bench_res = []
    global_metrics = {}
    config_breakdown = {c: {} for c in configs}
    all_provenance = []
    
    proc = psutil.Process()
    peak_ram = proc.memory_info().rss
    def update_ram():
        nonlocal peak_ram
        peak_ram = max(peak_ram, proc.memory_info().rss)

    # 1. Source scale space for dictionary building
    # Since we do Controlled Reconstruction, the "source" for 2x is 180x110. 
    # But wait, the prompt says "Each level must be produced directly from the input source". 
    # For a 2x enlargement of the 180x110 input, the input source is 180x110!
    # So we build a dictionary FROM the 180x110 input image for the 2x test.
    
    for sc in scales:
        s_n, factor = sc["n"], sc["factor"]
        s_w, s_h = w // factor, h // factor
        lr_srgb = cv2.resize(gt_arr, (s_w, s_h), interpolation=cv2.INTER_AREA)
        lr_lin = srgb_to_linear(lr_srgb)
        
        metrics_dict = {}
        
        # Bicubic
        st = time.time()
        bc = cv2.resize(lr_srgb, (w, h), interpolation=cv2.INTER_CUBIC)
        bc_rt = (time.time() - st) * 1000
        bc_m = compute_all_metrics(gt_arr, bc)
        bc_m["runtime_ms"] = bc_rt
        for r_name, (cx1, cy1, cx2, cy2) in crops_def.items():
             bc_m[f"region_{r_name}"] = compute_all_metrics(gt_arr[cy1:cy2, cx1:cx2], bc[cy1:cy2, cx1:cx2])
        metrics_dict["Bicubic"] = bc_m
        
        # Lanczos (Baseline)
        st = time.time()
        lz = cv2.resize(lr_srgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        lz_rt = (time.time() - st) * 1000
        lz_m = compute_all_metrics(gt_arr, lz)
        lz_m["runtime_ms"] = lz_rt
        for r_name, (cx1, cy1, cx2, cy2) in crops_def.items():
             lz_m[f"region_{r_name}"] = compute_all_metrics(gt_arr[cy1:cy2, cx1:cx2], lz[cy1:cy2, cx1:cx2])
        metrics_dict["Lanczos"] = lz_m
        
        recons = {"GT": Image.fromarray(gt_arr), "Bicubic": Image.fromarray(bc), "Lanczos": Image.fromarray(lz)}
        
        # Build scale space and dictionaries from the lr_lin image
        scale_levels = [1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25]
        pyramid = create_scale_space(lr_lin, scale_levels)
        
        dicts = {}
        for p_size in [5, 7]:
            d = PatchDictionary(patch_size=p_size, stride=1) # Extract dense for dict
            for slvl in scale_levels:
                high_img = pyramid[slvl]
                low_img = create_low_surrogate(high_img, downscale_factor=float(factor))
                d.add_scale_level(slvl, high_img, low_img)
            d.build_tree()
            dicts[p_size] = d
        
        update_ram()
        
        target_baseline = cv2.resize(lr_lin, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        for c_name, cfg in configs.items():
            st = time.time()
            d = dicts[cfg["patch_size"]]
            
            out_lin, final_res, conf_map, stats = reconstruct_overlap_add(target_baseline, d, cfg, scale_factor=float(factor), tile_size=256)
            out_srgb = linear_to_srgb(out_lin, to_uint8=True)
            rt = (time.time() - st) * 1000
            
            # Tiling equivalence check
            out_untiled, _, _, _ = reconstruct_overlap_add(target_baseline, d, cfg, scale_factor=float(factor), tile_size=9999)
            out_srgb_untiled = linear_to_srgb(out_untiled, to_uint8=True)
            te_max, te_mean = compute_tiling_equivalence_error(out_srgb_untiled, out_srgb)
            
            isepr_m = compute_all_metrics(gt_arr, out_srgb)
            isepr_m["runtime_ms"] = rt
            isepr_m["tiling_equivalence_max_error"] = te_max
            isepr_m["tiling_equivalence_mean_error"] = te_mean
            isepr_m["active_patches"] = stats["active_patches"]
            isepr_m["accepted_patches"] = stats["accepted_patches"]
            isepr_m["dictionary_size"] = len(d.descriptors)
            if stats["active_patches"] > 0:
                isepr_m["accepted_percent"] = (stats["accepted_patches"] / stats["active_patches"]) * 100.0
            else:
                isepr_m["accepted_percent"] = 0.0
                
            for r_name, (cx1, cy1, cx2, cy2) in crops_def.items():
                 isepr_m[f"region_{r_name}"] = compute_all_metrics(gt_arr[cy1:cy2, cx1:cx2], out_srgb[cy1:cy2, cx1:cx2])
                 
            metrics_dict[c_name] = isepr_m
            recons[c_name] = Image.fromarray(out_srgb)
            all_provenance.extend(stats["provenance"])
            update_ram()
            
        bench_res.append({"scale": s_n, "metrics": metrics_dict})
        global_metrics[s_n] = metrics_dict
        
        # Save contact sheet for this scale
        make_contact_sheet(recons, f"Recon {s_n}", os.path.join(out_dir, f"contact_recon_{s_n}_full.png"))
        for crop_name, (cx1, cy1, cx2, cy2) in crops_def.items():
            crop_recons = {k: v.crop((cx1, cy1, cx2, cy2)) for k, v in recons.items()}
            make_contact_sheet(crop_recons, f"{crop_name} {s_n}", os.path.join(out_dir, f"contact_recon_{s_n}_{crop_name}.png"))

    # Eligibility & Diagnostic Configuration
    eligible = []
    diagnostic_config = "ISEPR_A"
    best_f1_for_diag = -1
    
    for c_name in configs:
        reasons = []
        f1_improvements = 0
        avg_f1 = 0
        cond_psnr = True
        cond_ssim = True
        cond_ring = True
        cond_seam = True
        
        for s in scales:
            s_n = s["n"]
            lz_m = global_metrics[s_n]["Lanczos"]
            c_m = global_metrics[s_n][c_name]
            
            psnr_loss = lz_m["psnr"] - c_m["psnr"]
            ssim_loss = lz_m["ssim"] - c_m["ssim"]
            ring_diff = c_m["ringing_percent"] - lz_m["ringing_percent"]
            
            if psnr_loss > 0.15: cond_psnr = False
            if ssim_loss > 0.003: cond_ssim = False
            if ring_diff > 0.30: cond_ring = False
            if c_m["tiling_equivalence_max_error"] > 1.0: cond_seam = False
            
            if c_m["edge_f1"] > lz_m["edge_f1"]:
                f1_improvements += 1
            avg_f1 += c_m["edge_f1"]
            
        avg_f1 /= len(scales)
        if avg_f1 > best_f1_for_diag:
            best_f1_for_diag = avg_f1
            diagnostic_config = c_name
            
        if not cond_psnr: reasons.append("PSNR loss > 0.15dB")
        if not cond_ssim: reasons.append("SSIM loss > 0.003")
        if not cond_ring: reasons.append("Ringing > 0.30%")
        if not cond_seam: reasons.append("Tiling error > 1")
        if f1_improvements < 2: reasons.append("Edge F1 did not improve at >=2 scales")
        
        is_eligible = (cond_psnr and cond_ssim and cond_ring and cond_seam and f1_improvements >= 2)
        if is_eligible: eligible.append(c_name)
        
        config_breakdown[c_name] = {
            "eligible": is_eligible,
            "rejection_reasons": reasons,
            "edge_f1_improvements": f1_improvements
        }
        
    best_config = None
    if eligible:
        best_avg = -1
        for c in eligible:
            avg = np.mean([global_metrics[s["n"]][c]["edge_f1"] for s in scales])
            if avg > best_avg:
                best_avg = avg
                best_config = c
                
    status = "NEEDS_USER_VISUAL_CHECK" if best_config else "NO_MEASURABLE_GAIN"
    sel_config = best_config if best_config else "none"

    # Operational Enlargement
    enlarge_scales = [2, 4, 8]
    for factor in enlarge_scales:
        ew, eh = w * factor, h * factor
        lz_arr = cv2.resize(gt_arr, (ew, eh), interpolation=cv2.INTER_LANCZOS4)
        Image.fromarray(lz_arr).save(os.path.join(out_dir, f"op_{factor}x_Lanczos.png"))
        
        # Build dictionary from original gt_arr
        pyramid = create_scale_space(gt_lin, [1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25])
        cfg = configs[diagnostic_config]
        d = PatchDictionary(patch_size=cfg["patch_size"], stride=3)
        for slvl in [1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25]:
            hi = pyramid[slvl]
            lo = create_low_surrogate(hi, downscale_factor=float(factor))
            d.add_scale_level(slvl, hi, lo)
        d.build_tree()
        
        target_baseline = cv2.resize(gt_lin, (ew, eh), interpolation=cv2.INTER_LANCZOS4)
        out_lin, _, conf_map, _ = reconstruct_overlap_add(target_baseline, d, cfg, scale_factor=float(factor), tile_size=256)
        out_srgb = linear_to_srgb(out_lin, to_uint8=True)
        out_srgb = label_diagnostic(out_srgb)
        
        Image.fromarray(out_srgb).save(os.path.join(out_dir, f"op_{factor}x_ISEPR_diagnostic.png"))
        
        # Save map
        conf_vis = (conf_map * 255).astype(np.uint8)
        Image.fromarray(conf_vis).save(os.path.join(out_dir, f"op_{factor}x_conf_map.png"))
        
        op_crops = {
            "wheel": (185*factor, 100*factor, 260*factor, 205*factor),
            "headlamp": (135*factor, 70*factor, 195*factor, 125*factor),
            "plate": (25*factor, 125*factor, 100*factor, 165*factor)
        }
        
        imgs_full = {"Lanczos": Image.fromarray(lz_arr), f"ISEPR ({diagnostic_config})": Image.fromarray(out_srgb)}
        make_contact_sheet(imgs_full, "Full", os.path.join(out_dir, f"contact_op_{factor}x_full.png"), mark_diagnostic=True, diagnostic_key="ISEPR")
        
        for crop_name, (cx1, cy1, cx2, cy2) in op_crops.items():
            imgs_crop = {
                "Lanczos": Image.fromarray(lz_arr[cy1:cy2, cx1:cx2]),
                f"ISEPR ({diagnostic_config})": Image.fromarray(out_srgb[cy1:cy2, cx1:cx2])
            }
            make_contact_sheet(imgs_crop, crop_name, os.path.join(out_dir, f"contact_op_{factor}x_{crop_name}.png"), mark_diagnostic=True, diagnostic_key="ISEPR")
        update_ram()

    results_obj = {
        "status": status,
        "selected_config": sel_config,
        "diagnostic_config": diagnostic_config,
        "config_selection_breakdown": config_breakdown,
        "reconstruction_benchmark": bench_res
    }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
            
    with open(os.path.join(out_dir, "results.json"), "w", newline='\n') as f:
        json.dump(results_obj, f, indent=2, cls=NumpyEncoder)
        f.write('\n')
        
    with open(os.path.join(out_dir, "provenance.json"), "w", newline='\n') as f:
        json.dump(all_provenance, f, indent=2, cls=NumpyEncoder)
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
        "algorithm_version": "isepr-v1",
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
        f.write(json.dumps(ev_obj, cls=NumpyEncoder) + "\n")

    print(f"Status: {status}")

def main():
    try:
        run_benchmark()
    except Exception as e:
        out_dir = "m5c_results"
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
