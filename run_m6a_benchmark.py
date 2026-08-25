#!/usr/bin/env python3
"""
Milestone 6A: Classical Multi-Frame Super-Resolution — Controlled Benchmark
"""
import sys
import os
import json
import time
import uuid
import hashlib
import subprocess
import traceback
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from m6a_multiframe.forward_model import srgb_to_lin, lin_to_srgb, forward_model
from m6a_multiframe.frame_generator import generate_frames
from m6a_multiframe.registration import register_frames
from m6a_multiframe.robust_fusion import shift_and_add, robust_fusion
from m6a_multiframe.backprojection import iterative_backprojection
from m6a_multiframe.metrics import (
    compute_psnr, compute_ssim, edge_f1_score, compute_gradient_energy,
    compute_laplacian_variance, color_histogram_distance,
    calculate_ringing_percentage, compute_clipping_percentage,
    registration_rmse, compute_all_metrics
)

OUT_DIR = "m6a_results"
SEED = 42


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        status_output = subprocess.check_output(["git", "status", "--porcelain"]).decode()
        dirty = bool(status_output.strip())
        return commit, dirty, status_output
    except Exception:
        return "unknown", True, "unknown"


def get_dep_versions():
    return {
        'numpy': np.__version__,
        'opencv': cv2.__version__,
        'python': sys.version,
    }


def save_img(name, img_uint8):
    """Save uint8 image and return path."""
    path = os.path.join(OUT_DIR, name)
    Image.fromarray(img_uint8).save(path)
    return path


def lin_to_uint8(lin_img):
    return (lin_to_srgb(np.clip(lin_img, 0, 1)) * 255).astype(np.uint8)


def make_contact_sheet(images, labels, max_w=1800):
    """Create a labeled contact sheet from a list of images."""
    n = len(images)
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Resize all to same height
    target_h = images[0].shape[0]
    resized = []
    for img in images:
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        resized.append(img)

    label_h = 30
    sheets = []
    for img, label in zip(resized, labels):
        h, w = img.shape[:2]
        canvas = np.ones((h + label_h, w, 3), dtype=np.uint8) * 255
        canvas[label_h:, :, :] = img if img.ndim == 3 else np.stack([img]*3, axis=-1)
        cv2.putText(canvas, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        sheets.append(canvas)

    return np.hstack(sheets)


def run_experiment(gt_lin, scale_factor, num_frames, base_sigma, regions, experiment_name):
    """Run a single scale experiment. Returns dict of results."""
    print(f"\n=== {experiment_name}: {scale_factor}x, {num_frames} frames ===")
    result = {}
    artifacts = {}
    t_start = time.time()

    hr_h, hr_w = gt_lin.shape[:2]
    gt_uint8 = lin_to_uint8(gt_lin)

    # 1. Generate frames
    print("  Generating frames...")
    frames, true_translations = generate_frames(gt_lin, scale_factor, num_frames, base_sigma, seed=SEED)

    # Convert true translations from HR to LR space for comparison
    true_lr_translations = [(dx / scale_factor, dy / scale_factor) for dx, dy in true_translations]

    # Save all input frames
    for i, frame in enumerate(frames):
        fname = f"frame_{scale_factor}x_{i:02d}.png"
        save_img(fname, lin_to_uint8(frame))
        artifacts[fname] = {'type': 'input_frame'}

    # Save ground truth
    gt_name = f"ground_truth_{scale_factor}x.png"
    save_img(gt_name, gt_uint8)
    artifacts[gt_name] = {'type': 'ground_truth'}

    # 2. Reconstruct using clean API (WITHOUT ground truth)
    print("  Reconstructing...")
    from m6a_multiframe.pipeline import reconstruct_from_frames
    config = {
        'ref_index': 0,
        'cc_threshold': 0.85,
        'base_sigma': base_sigma,
        'max_iters': 5,
        'initial_step': 0.1,
        'reg_weight': 0.15
    }
    
    recon_out = reconstruct_from_frames(frames, scale_factor, config)
    est_translations = recon_out['estimated_translations']
    accepted_ids = recon_out['accepted_ids']
    rejected_ids = [i for i in range(num_frames) if i not in accepted_ids]
    
    print(f"  Accepted: {len(accepted_ids)}, Rejected: {len(rejected_ids)}")

    # Registration error
    reg_rmse = registration_rmse(est_translations, true_lr_translations)
    print(f"  Registration RMSE: {reg_rmse:.4f} LR pixels")

    result['registration'] = {
        'estimated_translations': est_translations,
        'true_translations': true_lr_translations,
        'accepted_ids': accepted_ids,
        'rejected_ids': rejected_ids,
        'registration_rmse': reg_rmse,
    }

    # Extract outputs
    ibp_result = recon_out['reconstruction']
    coverage = recon_out['coverage']
    confidence = recon_out['confidence']
    ibp_history = recon_out['history']

    # 3. Baselines (Lanczos and Shift-and-Add)
    print("  Computing baselines...")
    lanczos_lin = cv2.resize(frames[0], (hr_w, hr_h), interpolation=cv2.INTER_LANCZOS4)
    lanczos_uint8 = lin_to_uint8(lanczos_lin)
    save_img(f"lanczos_{scale_factor}x.png", lanczos_uint8)
    artifacts[f"lanczos_{scale_factor}x.png"] = {'type': 'baseline'}

    acc_frames = [frames[i] for i in accepted_ids]
    acc_est = [est_translations[i] for i in accepted_ids]
    
    saa_lin = shift_and_add(acc_frames, acc_est, scale_factor)
    saa_uint8 = lin_to_uint8(saa_lin)
    save_img(f"shift_and_add_{scale_factor}x.png", saa_uint8)
    artifacts[f"shift_and_add_{scale_factor}x.png"] = {'type': 'shift_and_add'}

    # Re-compute robust fusion just for saving baseline (or skip saving it if not strictly required, but the metrics compute it)
    fused_lin, _, _ = robust_fusion(acc_frames, acc_est, scale_factor)
    fused_uint8 = lin_to_uint8(fused_lin)
    save_img(f"robust_fusion_{scale_factor}x.png", fused_uint8)
    artifacts[f"robust_fusion_{scale_factor}x.png"] = {'type': 'robust_fusion'}

    # Save coverage and confidence maps
    cov_vis = (coverage * 255).astype(np.uint8)
    save_img(f"coverage_{scale_factor}x.png", cv2.applyColorMap(cov_vis, cv2.COLORMAP_JET))
    artifacts[f"coverage_{scale_factor}x.png"] = {'type': 'coverage_map'}

    conf_vis = (confidence * 255).astype(np.uint8)
    save_img(f"confidence_{scale_factor}x.png", cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET))
    artifacts[f"confidence_{scale_factor}x.png"] = {'type': 'confidence_map'}

    result['coverage_percent'] = float(np.mean(coverage > 0.5) * 100)

    # Save final IB result
    ibp_uint8 = lin_to_uint8(ibp_result)
    save_img(f"multiframe_sr_{scale_factor}x.png", ibp_uint8)
    artifacts[f"multiframe_sr_{scale_factor}x.png"] = {'type': 'multiframe_sr'}

    # 6. Compute metrics for all methods
    print("  Computing metrics...")
    methods = {
        'Lanczos': lanczos_uint8,
        'ShiftAndAdd': saa_uint8,
        'RobustFusion': fused_uint8,
        'MultiframeSR': ibp_uint8,
    }

    all_metrics = {}
    for mname, pred in methods.items():
        # Ensure dimensions match
        ph, pw = pred.shape[:2]
        gh, gw = gt_uint8.shape[:2]
        if ph != gh or pw != gw:
            pred = cv2.resize(pred, (gw, gh), interpolation=cv2.INTER_LANCZOS4)
        m = compute_all_metrics(gt_uint8, pred)
        # Regional metrics
        for rname, (y0, y1, x0, x1) in regions.items():
            rm = compute_all_metrics(gt_uint8[y0:y1, x0:x1], pred[y0:y1, x0:x1])
            m[f'region_{rname}'] = rm
        all_metrics[mname] = m

    result['metrics'] = all_metrics

    # Source consistency for MultiframeSR
    simulated = forward_model(ibp_result, scale_factor, base_sigma * scale_factor, (0, 0))
    fh, fw = frames[0].shape[:2]
    sh, sw = simulated.shape[:2]
    mh, mw = min(fh, sh), min(fw, sw)
    src_err = float(np.sqrt(np.mean((frames[0][:mh, :mw] - simulated[:mh, :mw]) ** 2)))
    result['metrics']['MultiframeSR']['source_consistency_error'] = src_err

    # 7. Residual visualizations
    for mname, pred in methods.items():
        if mname == 'Lanczos':
            continue
        ph, pw = pred.shape[:2]
        gh, gw = gt_uint8.shape[:2]
        if ph != gh or pw != gw:
            pred = cv2.resize(pred, (gw, gh), interpolation=cv2.INTER_LANCZOS4)
        residual = gt_uint8.astype(np.float32) - pred.astype(np.float32)
        res_vis = np.clip((residual / 2 + 128), 0, 255).astype(np.uint8)
        fname = f"residual_{mname}_{scale_factor}x.png"
        save_img(fname, res_vis)
        artifacts[fname] = {'type': 'residual'}

    # 8. Contact sheets
    print("  Creating contact sheets...")
    contact = make_contact_sheet(
        [gt_uint8, lanczos_uint8, saa_uint8, fused_uint8, ibp_uint8],
        ["Ground Truth", "Lanczos", "Shift+Add", "Robust Fusion", "Multi-Frame SR"]
    )
    save_img(f"contact_sheet_{scale_factor}x.png", contact)
    artifacts[f"contact_sheet_{scale_factor}x.png"] = {'type': 'contact_sheet'}

    # Regional contact sheets
    for rname, (y0, y1, x0, x1) in regions.items():
        imgs = [gt_uint8[y0:y1, x0:x1]]
        lbls = ["GT"]
        for mname, pred in methods.items():
            ph, pw = pred.shape[:2]
            gh, gw = gt_uint8.shape[:2]
            if ph != gh or pw != gw:
                pred = cv2.resize(pred, (gw, gh), interpolation=cv2.INTER_LANCZOS4)
            imgs.append(pred[y0:y1, x0:x1])
            lbls.append(mname)
        rc = make_contact_sheet(imgs, lbls)
        fname = f"contact_{rname}_{scale_factor}x.png"
        save_img(fname, rc)
        artifacts[fname] = {'type': f'contact_{rname}'}

    # Registration error plot (simple cv2-based visualization)
    reg_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    for i, ((ex, ey), (tx, ty)) in enumerate(zip(est_translations, true_lr_translations)):
        # Plot true (blue) and estimated (red) translations
        cx, cy = 200, 150
        sc = 30  # scale for visualization
        cv2.circle(reg_img, (int(cx + tx * sc), int(cy + ty * sc)), 4, (255, 0, 0), -1)
        cv2.circle(reg_img, (int(cx + ex * sc), int(cy + ey * sc)), 4, (0, 0, 255), -1)
        cv2.line(reg_img,
                 (int(cx + tx * sc), int(cy + ty * sc)),
                 (int(cx + ex * sc), int(cy + ey * sc)),
                 (0, 200, 0), 1)
    cv2.putText(reg_img, "Blue=True Red=Est", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(reg_img, f"RMSE={reg_rmse:.4f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    save_img(f"registration_plot_{scale_factor}x.png", reg_img)
    artifacts[f"registration_plot_{scale_factor}x.png"] = {'type': 'registration_plot'}

    # Accepted/rejected visualization
    ar_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    for i in range(num_frames):
        color = (0, 180, 0) if i in accepted_ids else (0, 0, 220)
        x = 20 + (i % 10) * 35
        y = 40 + (i // 10) * 60
        cv2.rectangle(ar_img, (x, y), (x + 25, y + 25), color, -1)
        cv2.putText(ar_img, str(i), (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(ar_img, "Green=Accepted Red=Rejected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    save_img(f"accepted_rejected_{scale_factor}x.png", ar_img)
    artifacts[f"accepted_rejected_{scale_factor}x.png"] = {'type': 'accepted_rejected'}

    runtime = time.time() - t_start
    result['runtime_seconds'] = runtime
    result['artifacts'] = artifacts
    result['ibp_history'] = ibp_history

    # Objective history plot
    h_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    if ibp_history:
        iters = [h['iteration'] for h in ibp_history]
        objs = [h['objective'] for h in ibp_history]
        if len(iters) > 1:
            max_obj, min_obj = max(objs), min(objs)
            for j in range(len(iters) - 1):
                x1 = int(10 + iters[j] / (max(iters) + 1e-6) * 380)
                y1 = int((1 - (objs[j] - min_obj) / (max_obj - min_obj + 1e-6)) * 270 + 10)
                x2 = int(10 + iters[j+1] / (max(iters) + 1e-6) * 380)
                y2 = int((1 - (objs[j+1] - min_obj) / (max_obj - min_obj + 1e-6)) * 270 + 10)
                cv2.line(h_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(h_img, f"Objective {scale_factor}x", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    save_img(f"objective_history_{scale_factor}x.png", h_img)
    artifacts[f"objective_history_{scale_factor}x.png"] = {'type': 'objective_history'}

    print(f"  Done in {runtime:.1f}s")
    return result


def determine_status(results):
    """Determine acceptance status from results."""
    r2 = results.get('2x', {}).get('metrics', {})
    r4 = results.get('4x', {}).get('metrics', {})

    if not r2 or not r4:
        return 'BLOCKED', ['Missing experiment results']

    reasons = []
    lanczos_2x = r2.get('Lanczos', {})
    mfsr_2x = r2.get('MultiframeSR', {})
    lanczos_4x = r4.get('Lanczos', {})
    mfsr_4x = r4.get('MultiframeSR', {})

    # PSNR gain
    psnr_gain_2x = mfsr_2x.get('psnr', 0) - lanczos_2x.get('psnr', 0)
    psnr_gain_4x = mfsr_4x.get('psnr', 0) - lanczos_4x.get('psnr', 0)
    ssim_gain_2x = mfsr_2x.get('ssim', 0) - lanczos_2x.get('ssim', 0)
    ssim_gain_4x = mfsr_4x.get('ssim', 0) - lanczos_4x.get('ssim', 0)

    if psnr_gain_2x < 0.50:
        reasons.append(f"2x PSNR gain {psnr_gain_2x:.3f} dB < 0.50 dB")
    if ssim_gain_2x < 0.005:
        reasons.append(f"2x SSIM gain {ssim_gain_2x:.4f} < 0.005")
    if psnr_gain_4x < 0.25:
        reasons.append(f"4x PSNR gain {psnr_gain_4x:.3f} dB < 0.25 dB")
    if ssim_gain_4x < 0.003:
        reasons.append(f"4x SSIM gain {ssim_gain_4x:.4f} < 0.003")

    # Edge F1
    ef1_2x_gain = mfsr_2x.get('edge_f1', 0) - lanczos_2x.get('edge_f1', 0)
    ef1_4x_gain = mfsr_4x.get('edge_f1', 0) - lanczos_4x.get('edge_f1', 0)
    if ef1_2x_gain <= 0:
        reasons.append(f"2x edge F1 not improved ({ef1_2x_gain:.4f})")
    if ef1_4x_gain <= 0:
        reasons.append(f"4x edge F1 not improved ({ef1_4x_gain:.4f})")

    # Ringing
    ring_2x = mfsr_2x.get('ringing_percent', 0) - lanczos_2x.get('ringing_percent', 0)
    ring_4x = mfsr_4x.get('ringing_percent', 0) - lanczos_4x.get('ringing_percent', 0)
    if ring_2x > 0.20:
        reasons.append(f"2x ringing increase {ring_2x:.2f}% > 0.20%")
    if ring_4x > 0.20:
        reasons.append(f"4x ringing increase {ring_4x:.2f}% > 0.20%")

    # Registration RMSE
    reg_2x = results['2x']['registration']['registration_rmse']
    reg_4x = results['4x']['registration']['registration_rmse']
    if reg_2x > 0.25:
        reasons.append(f"2x registration RMSE {reg_2x:.4f} > 0.25")
    if reg_4x > 0.25:
        reasons.append(f"4x registration RMSE {reg_4x:.4f} > 0.25")

    if reasons:
        return 'NO_MEASURABLE_GAIN', reasons
    return 'CONTROLLED_GAIN_CONFIRMED', []


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    run_id = str(uuid.uuid4())
    utc_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    git_commit, git_dirty, git_status = get_git_info()
    command = " ".join(sys.argv)
    
    print(f"Run ID: {run_id}")
    print(f"UTC: {utc_time}")
    print(f"Commit: {git_commit}")

    # Load ground truth
    gt_path = "m4d_test_assets/porche.png"
    gt_img = np.array(Image.open(gt_path).convert("RGB"))
    gt_lin = srgb_to_lin(gt_img)
    source_sha256 = sha256_file(gt_path)

    print(f"Ground truth: {gt_img.shape} -> linear {gt_lin.shape}")

    # Regions for evaluation (in HR coordinates)
    regions = {
        'wheel': (100, 205, 185, 260),
        'headlamp': (70, 125, 135, 195),
        'plate': (125, 165, 25, 100),
    }

    results = {}
    peak_ram = 0

    # 2x experiment: 8 frames
    try:
        import tracemalloc
        tracemalloc.start()
        results['2x'] = run_experiment(gt_lin, 2, 8, 0.5, regions, "2x Experiment")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results['2x']['peak_ram_mb'] = peak / 1024 / 1024
    except Exception as e:
        results['2x'] = {'error': str(e), 'traceback': traceback.format_exc()}

    # 4x experiment: 16 frames
    try:
        import tracemalloc
        tracemalloc.start()
        results['4x'] = run_experiment(gt_lin, 4, 16, 0.5, regions, "4x Experiment")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results['4x']['peak_ram_mb'] = peak / 1024 / 1024
    except Exception as e:
        results['4x'] = {'error': str(e), 'traceback': traceback.format_exc()}

    # Determine status
    status, rejection_reasons = determine_status(results)
    results['status'] = status
    results['rejection_reasons'] = rejection_reasons

    print(f"\n=== STATUS: {status} ===")
    for r in rejection_reasons:
        print(f"  - {r}")

    # Save results JSON
    results_json = {
        'status': status,
        'rejection_reasons': rejection_reasons,
        'experiments': {}
    }
    for scale in ['2x', '4x']:
        if scale in results and 'metrics' in results[scale]:
            exp = {
                'metrics': results[scale]['metrics'],
                'registration': results[scale]['registration'],
                'coverage_percent': results[scale].get('coverage_percent', 0),
                'runtime_seconds': results[scale].get('runtime_seconds', 0),
                'peak_ram_mb': results[scale].get('peak_ram_mb', 0),
            }
            results_json['experiments'][scale] = exp

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    # Evidence JSONL
    evidence = {
        'run_uuid': run_id,
        'utc_time': utc_time,
        'command': command,
        'git_commit': git_commit,
        'git_dirty_at_start': git_dirty,
        'git_status_at_start': git_status,
        'source_sha256': source_sha256,
        'dependency_versions': get_dep_versions(),
        'random_seeds': [SEED],
    }

    with open(os.path.join(OUT_DIR, 'evidence.jsonl'), 'a') as f:
        for scale in ['2x', '4x']:
            if scale in results:
                r = results[scale]
                ev = {
                    **evidence,
                    'scale_factor': int(scale[0]),
                    'frame_count': 8 if scale == '2x' else 16,
                }
                
                if 'error' in r:
                    ev['exception'] = r['error']
                    ev['traceback'] = r['traceback']
                else:
                    ev['estimated_translations'] = r['registration']['estimated_translations']
                    ev['true_translations'] = r['registration']['true_translations']
                    ev['accepted_frames'] = r['registration']['accepted_ids']
                    ev['rejected_frames'] = r['registration']['rejected_ids']
                    ev['registration_error'] = r['registration']['registration_rmse']
                    ev['runtime_seconds'] = r.get('runtime_seconds', 0)
                    ev['peak_ram_mb'] = r.get('peak_ram_mb', 0)
                    ev['output_dimensions'] = f"{360}x{220}"
                    
                    artifacts_list = []
                    for aname in r.get('artifacts', {}):
                        apath = os.path.join(OUT_DIR, aname)
                        if os.path.exists(apath):
                            artifacts_list.append({
                                'path': aname,
                                'sha256': sha256_file(apath),
                                'size': os.path.getsize(apath),
                            })
                    ev['artifacts'] = artifacts_list
                    
                f.write(json.dumps(ev, default=str) + '\n')

    # Provenance
    with open(os.path.join(OUT_DIR, 'provenance.json'), 'w') as f:
        json.dump({
            'git_commit': git_commit,
            'utc_time': utc_time,
            'run_uuid': run_id,
        }, f, indent=2)

    print(f"\nAll results saved to {OUT_DIR}/")
    return status


if __name__ == '__main__':
    main()
