import os
import argparse
import json
import sys
import time
import numpy as np
import cv2
from PIL import Image

os.environ["HF_HOME"] = r"E:\SFI\hf_cache"

# Ensure scalefield_lite modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))
from scalefield_lite.coordinates import calculate_zoom_levels
from scalefield_lite.seeds import calculate_sha256, generate_seed
from scalefield_lite.source_sampler import sample_and_resize
from scalefield_lite.generator import ScaleFieldGenerator
from scalefield_lite.frequency_anchor import apply_frequency_anchor
from scalefield_lite.metrics import compute_gradient_energy, compute_laplacian_variance, compute_color_histogram_distance, compute_ssim, compute_edge_consistency
from scalefield_lite.evidence import record_evidence

def create_contact_sheet(images: list, output_path: str):
    w, h = images[0].size
    sheet = Image.new("RGB", (w * len(images), h))
    for i, img in enumerate(images):
        sheet.paste(img, (i * w, 0))
    sheet.save(output_path)

def align_and_compute_ssim(parent_img: Image.Image, child_img: Image.Image, zoom_ratio: float) -> float:
    """Downsample child to match parent center and compute SSIM."""
    if zoom_ratio <= 1.0:
        return 1.0
    
    parent_arr = np.array(parent_img)
    child_arr = np.array(child_img)
    
    h, w, c = parent_arr.shape
    crop_w = int(w / zoom_ratio)
    crop_h = int(h / zoom_ratio)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    parent_center = parent_arr[start_y:start_y+crop_h, start_x:start_x+crop_w]
    child_downsampled = cv2.resize(child_arr, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    
    return compute_ssim(parent_center, child_downsampled)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-description", type=str, default="A highly detailed close-up texture of a sports car front wheel rim and tire, retaining exact physical properties.")
    args = parser.parse_args()
    
    out_dir = "m4c_results"
    os.makedirs(out_dir, exist_ok=True)
    
    source_img_path = "porche.png"
    if not os.path.exists(source_img_path):
        print("Missing porche.png")
        sys.exit(1)
        
    source_sha = calculate_sha256(source_img_path)
    
    # Target Region: Front wheel/rim (approximate coordinates for a 1024x1024 Porsche image)
    # Even if 360x220, 0.35x, 0.70y is a good spot for the wheel.
    target_cx = 0.35
    target_cy = 0.70
    zoom_factors = [1.0, 4.0, 16.0, 64.0, 256.0]
    
    with Image.open(source_img_path) as img:
        w, h = img.size
        
    levels = calculate_zoom_levels(w, h, target_cx, target_cy, zoom_factors)
    
    try:
        generator = ScaleFieldGenerator()
    except Exception as e:
        record_evidence(
            os.path.join(out_dir, "evidence.jsonl"),
            sys.argv, source_sha, "stabilityai/sd-turbo", "unknown", [], [], str(e)
        )
        print("Failed to initialize generator:", e)
        sys.exit(1)
        
    levels_data = []
    artifacts = []
    
    parent_img = None
    final_images = []
    
    for lvl in levels:
        print(f"--- Level {lvl.level} (Zoom {lvl.zoom_factor}x) ---")
        start_time = time.time()
        
        # 1. Source Conditioning
        source_crop = sample_and_resize(source_img_path, lvl.box_pixels)
        source_crop_path = os.path.join(out_dir, f"level_{lvl.level}_source.png")
        source_crop.save(source_crop_path)
        artifacts.append({"path": source_crop_path, "sha256": calculate_sha256(source_crop_path)})
        
        # 2. Deterministic Seed
        seed = generate_seed(source_sha, lvl.box_normalized, lvl.zoom_factor, version="v1.0")
        print(f"Seed: {seed}")
        
        # 3. Independent Raw Generation
        raw_gen = generator.generate(source_crop, args.region_description, seed)
        raw_path = os.path.join(out_dir, f"level_{lvl.level}_raw.png")
        raw_gen.save(raw_path)
        artifacts.append({"path": raw_path, "sha256": calculate_sha256(raw_path)})
        
        # 4. Cross-scale Frequency Anchoring
        if parent_img is not None:
            anchored = apply_frequency_anchor(parent_img, raw_gen, zoom_ratio=4.0)
        else:
            anchored = raw_gen.copy()
            
        anchored_path = os.path.join(out_dir, f"level_{lvl.level}_anchored.png")
        anchored.save(anchored_path)
        artifacts.append({"path": anchored_path, "sha256": calculate_sha256(anchored_path)})
        
        final_images.append(anchored)
        
        end_time = time.time()
        
        # 5. Metrics
        raw_arr = np.array(raw_gen)
        anchored_arr = np.array(anchored)
        
        metrics = {
            "runtime_seconds": end_time - start_time,
            "gradient_energy_raw": compute_gradient_energy(raw_arr),
            "gradient_energy_anchored": compute_gradient_energy(anchored_arr),
            "laplacian_variance_raw": compute_laplacian_variance(raw_arr),
            "laplacian_variance_anchored": compute_laplacian_variance(anchored_arr),
            "color_hist_distance_raw_vs_anchored": compute_color_histogram_distance(raw_arr, anchored_arr),
            "ssim_raw_vs_anchored": compute_ssim(raw_arr, anchored_arr),
            "edge_consistency_raw_vs_anchored": compute_edge_consistency(raw_arr, anchored_arr),
            "ssim_parent_vs_anchored_child": align_and_compute_ssim(parent_img, anchored, zoom_ratio=4.0) if parent_img is not None else 1.0
        }
        print(f"Metrics: {metrics}")
        
        levels_data.append({
            "level": lvl.level,
            "zoom_factor": lvl.zoom_factor,
            "box_normalized": lvl.box_normalized,
            "seed": seed,
            "metrics": metrics
        })
        
        parent_img = anchored
        
    contact_sheet_path = os.path.join(out_dir, "contact_sheet.png")
    create_contact_sheet(final_images, contact_sheet_path)
    artifacts.append({"path": contact_sheet_path, "sha256": calculate_sha256(contact_sheet_path)})
    
    # Save results.json
    results_obj = {
        "overall_status": "NEEDS_USER_VISUAL_CHECK",
        "levels": levels_data
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_obj, f, indent=2)
    artifacts.append({"path": results_path, "sha256": calculate_sha256(results_path)})
    
    # Save evidence
    record_evidence(
        os.path.join(out_dir, "evidence.jsonl"),
        sys.argv, source_sha, generator.model_id, generator.revision, levels_data, artifacts
    )

if __name__ == "__main__":
    main()
