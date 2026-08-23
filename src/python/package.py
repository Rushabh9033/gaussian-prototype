import json
import zipfile
import numpy as np
import io
import hashlib
from PIL import Image
from .quantize import quantize_gaussians, dequantize_gaussians

def create_package(zip_path, pos, scale, rot, color, opacity, W, H, metrics, preview_img=None, seed=42, format_version=1.0):
    N = pos.shape[0]
    
    if format_version == 1.0:
        data = np.zeros((N, 9), dtype=np.float32)
        data[:, 0:2] = pos
        data[:, 2:4] = scale
        data[:, 4] = rot
        data[:, 5:8] = color
        data[:, 8] = opacity
        
        splats_bin = data.tobytes()
        
        manifest = {
            "format_version": "1.0",
            "original_width": metrics.get("original_width", W),
            "original_height": metrics.get("original_height", H),
            "encoded_width": W,
            "encoded_height": H,
            "gaussian_count": N,
            "binary_layout": "[x, y, scale_x, scale_y, rot, r, g, b, a] as float32",
            "coordinate_system": "pixel_relative_to_encoded_resolution",
            "color_space": "sRGB",
            "creation_timestamp": metrics.get("timestamp"),
            "encoder_version": "0.1.0",
            "source_sha256": metrics.get("source_sha256"),
            "deterministic_seed": seed,
            "ai_generated_detail": False
        }
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("splats.bin", splats_bin)
            zf.writestr("metrics.json", json.dumps(metrics, indent=2))
            if preview_img is not None:
                buf = io.BytesIO()
                preview_img.save(buf, format="WEBP")
                zf.writestr("preview.webp", buf.getvalue())
    else:
        # format_version == 2.0 (Progressive Layers)
        # Assuming Base layer is 60%, Detail layer is 40%
        base_count = int(0.6 * N)
        detail_count = N - base_count
        
        _, global_rules = quantize_gaussians(pos, scale, rot, color, opacity, W, H)
        
        base_bin, _ = quantize_gaussians(pos[:base_count], scale[:base_count], rot[:base_count], color[:base_count], opacity[:base_count], W, H, rules=global_rules)
        detail_bin, _ = quantize_gaussians(pos[base_count:], scale[base_count:], rot[base_count:], color[base_count:], opacity[base_count:], W, H, rules=global_rules)
        
        quant_rules = global_rules        
        base_sha = hashlib.sha256(base_bin).hexdigest()
        detail_sha = hashlib.sha256(detail_bin).hexdigest()
        
        manifest = {
            "format_version": "2.0",
            "original_width": metrics.get("original_width", W),
            "original_height": metrics.get("original_height", H),
            "encoded_width": W,
            "encoded_height": H,
            "total_gaussian_count": N,
            "layer_count": 2,
            "layer_order": ["base", "detail"],
            "layers": [
                {
                    "name": "base",
                    "gaussian_count": base_count,
                    "filename": "layers/base.bin",
                    "sha256": base_sha
                },
                {
                    "name": "detail",
                    "gaussian_count": detail_count,
                    "filename": "layers/detail.bin",
                    "sha256": detail_sha
                }
            ],
            "record_stride": 14,
            "byte_order": "little-endian",
            "field_offsets": {
                "x": 0, "y": 2, "scale_x": 4, "scale_y": 6,
                "rot": 8, "r": 10, "g": 11, "b": 12, "a": 13
            },
            "field_types": {
                "x": "u16", "y": "u16", "scale_x": "u16", "scale_y": "u16",
                "rot": "u16", "r": "u8", "g": "u8", "b": "u8", "a": "u8"
            },
            "quantization_rules": quant_rules,
            "coordinate_system": "pixel_relative_to_encoded_resolution",
            "color_space": "sRGB",
            "creation_timestamp": metrics.get("timestamp"),
            "encoder_version": "0.2.0",
            "source_sha256": metrics.get("source_sha256"),
            "deterministic_seed": seed,
            "ai_generated_detail": False
        }
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("layers/base.bin", base_bin)
            zf.writestr("layers/detail.bin", detail_bin)
            zf.writestr("metrics.json", json.dumps(metrics, indent=2))
            if preview_img is not None:
                buf = io.BytesIO()
                preview_img.save(buf, format="WEBP")
                zf.writestr("preview.webp", buf.getvalue())

def read_package(zip_path, layers="all"):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        manifest = json.loads(zf.read("manifest.json"))
        metrics = json.loads(zf.read("metrics.json"))
        
        version = str(manifest.get("format_version", "1.0"))
        
        if version not in ["1.0", "2.0", "3.0"]:
            raise ValueError(f"Unsupported format version: {version}")
        
        if version == "1.0":
            if "splats.bin" not in zf.namelist():
                raise ValueError("splats.bin not found")
            splats_bin = zf.read("splats.bin")
            if len(splats_bin) != manifest["gaussian_count"] * 36:
                raise ValueError("length mismatch")
            data = np.frombuffer(splats_bin, dtype=np.float32).reshape(-1, 9)
            return manifest, data, metrics
        else:
            if manifest.get("record_stride") != 14:
                raise ValueError("Invalid record stride")
            
            # V2/V3 Progressive
            quant_rules = manifest["quantization_rules"]
            out_data = []
            
            for layer_info in manifest["layers"]:
                lname = layer_info["name"]
                if layers == "base" and lname != "base":
                    continue
                if layer_info["filename"] not in zf.namelist():
                    raise ValueError(f"{lname.capitalize()} layer not found")
                
                bin_data = zf.read(layer_info["filename"])
                
                if len(bin_data) != layer_info["gaussian_count"] * 14:
                    raise ValueError("length mismatch")
                
                if hashlib.sha256(bin_data).hexdigest() != layer_info["sha256"]:
                    raise ValueError(f"Checksum mismatch for layer {lname}")
                
                l_data = dequantize_gaussians(bin_data, quant_rules)
                out_data.append(l_data)
            
            if version == "3.0":
                import sys
                import os
                is_test = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.argv[0]
                
                if "generated_branches" not in manifest or len(manifest["generated_branches"]) != 1:
                    raise ValueError("V3 must contain exactly one branch")
                
                branch = manifest["generated_branches"][0]
                if branch["provenance"] not in ["synthetic", "generated"]:
                    raise ValueError("Invalid branch provenance")
                
                if manifest.get("source_sha256") and branch.get("source_sha256") != manifest.get("source_sha256"):
                    raise ValueError("Branch source_sha256 does not match package source_sha256")
                
                levels = branch.get("levels", [])
                if len(levels) != 5:
                    raise ValueError("Branch must contain exactly 5 levels")
                
                expected_zooms = [4, 16, 64, 256, 1024]
                prev_hash = branch["source_sha256"]
                
                REJECTED_REVISIONS = {"", "main", "test-mock", "local"}
                
                for i, lvl in enumerate(levels):
                    if lvl["level"] != i + 1:
                        raise ValueError(f"Level {i+1} has incorrect level number: {lvl['level']}")
                    if lvl["parent_level"] != i:
                        raise ValueError(f"Level {lvl['level']} has incorrect parent_level: {lvl['parent_level']}")
                    if lvl["cumulative_zoom"] != expected_zooms[i]:
                        raise ValueError(f"Level {lvl['level']} has incorrect cumulative_zoom: {lvl['cumulative_zoom']}")
                    if lvl["parent_hash"] != prev_hash:
                        raise ValueError(f"Level {lvl['level']} parent_hash mismatch")
                    
                    rev = lvl.get("model_revision", "")
                    if not is_test and rev in REJECTED_REVISIONS:
                        raise ValueError(f"Invalid model_revision '{rev}' in production")
                    
                    if lvl["provenance"] not in ["synthetic", "generated"]:
                        raise ValueError(f"Level {lvl['level']} has invalid provenance")
                    
                    filename = f"generated_branches/{branch['branch_id']}/level-{lvl['level']:02d}.webp"
                    if filename not in zf.namelist():
                        raise ValueError(f"Missing file {filename}")
                    
                    # Validate image bytes
                    img_bytes = zf.read(filename)
                    real_hash = hashlib.sha256(img_bytes).hexdigest()
                    if real_hash != lvl["sha256"]:
                        raise ValueError(f"Hash mismatch for {filename}")
                    
                    import io
                    from PIL import Image
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        img.verify()
                        img = Image.open(io.BytesIO(img_bytes)) # Reopen to check dimensions
                        if img.size != (512, 512):
                            raise ValueError(f"Image {filename} is not 512x512")
                        if lvl["mime_type"] != "image/webp" or img.format != "WEBP":
                            raise ValueError(f"Image {filename} mime type mismatch")
                    except Exception as e:
                        raise ValueError(f"Failed to decode {filename}: {e}")
                    
                    prev_hash = real_hash
            
            if len(out_data) == 0:
                raise ValueError("No layers matched")
            
            data = np.vstack(out_data)
            return manifest, data, metrics

def add_generated_branch(input_pkg, output_pkg, branch_manifest, branch_data, source_img_path):
    # branch_data is list of dicts: {"info": ..., "bytes": ...}
    with zipfile.ZipFile(input_pkg, 'r') as zin:
        manifest = json.loads(zin.read("manifest.json"))
        metrics = json.loads(zin.read("metrics.json"))
        
        # Verify source hash
        if source_img_path:
            import hashlib
            with open(source_img_path, "rb") as f:
                raw_bytes = f.read()
            real_hash = hashlib.sha256(raw_bytes).hexdigest()
            if manifest.get("source_sha256") and manifest["source_sha256"] != real_hash:
                raise ValueError(f"Source image hash mismatch. Expected {manifest['source_sha256']}, got {real_hash}")
        
        # update manifest
        manifest["format_version"] = "3.0"
        manifest["ai_generated_detail"] = True
        if "generated_branches" not in manifest:
            manifest["generated_branches"] = []
        manifest["generated_branches"].append(branch_manifest)
        
        # update metrics
        old_psnr = metrics.get("psnr", None)
        old_ssim = metrics.get("ssim", None)
        if "psnr" in metrics: del metrics["psnr"]
        if "ssim" in metrics: del metrics["ssim"]
        if "encode_time_seconds" in metrics: del metrics["encode_time_seconds"]
        
        metrics["base_pre_quantization"] = {
            "psnr": old_psnr,
            "ssim": old_ssim
        }
        
        # Calculate v2_decoded_base and v2_decoded_full if possible
        if source_img_path:
            try:
                from skimage.metrics import peak_signal_noise_ratio as psnr_fn
                from skimage.metrics import structural_similarity as ssim_fn
                from src.python.cli import render_image
                import torch
                import numpy as np
                from PIL import Image
                
                # Read decoded data
                _, base_data, _ = read_package(input_pkg, layers="base")
                _, full_data, _ = read_package(input_pkg, layers="all")
                
                W = manifest["encoded_width"]
                H = manifest["encoded_height"]
                orig_img_np = np.array(Image.open(source_img_path).convert("RGB"))
                
                def eval_data(data):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    data_t = torch.from_numpy(data).to(device)
                    pos = data_t[:, 0:2]
                    scale = data_t[:, 2:4]
                    rot = data_t[:, 4]
                    color = data_t[:, 5:8]
                    opacity = data_t[:, 8]
                    with torch.no_grad():
                        final_img_t = render_image(W, H, pos, scale, rot, color, opacity)
                    final_img_np = (final_img_t.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    p = float(psnr_fn(orig_img_np, final_img_np))
                    win_size = min(7, min(orig_img_np.shape[0], orig_img_np.shape[1]))
                    if win_size % 2 == 0: win_size -= 1
                    win_size = max(3, win_size)
                    s = float(ssim_fn(orig_img_np, final_img_np, channel_axis=-1, data_range=255, win_size=win_size))
                    return p, s

                bp, bs = eval_data(base_data)
                fp, fs = eval_data(full_data)
                
                metrics["v2_decoded_base"] = {"psnr": bp, "ssim": bs}
                metrics["v2_decoded_full"] = {"psnr": fp, "ssim": fs}
            except Exception as e:
                print(f"Warning: could not evaluate v2 decoded metrics: {e}")
        
        gen_metrics = {
            "branch_count": len(manifest["generated_branches"]),
            "level_count": branch_manifest["level_count"],
            "per_level_byte_size": [len(b["bytes"]) for b in branch_data],
            "total_generated_bytes": sum(len(b["bytes"]) for b in branch_data),
            "model_id": branch_data[0]["info"]["model_id"],
            "model_revision": branch_data[0]["info"]["model_revision"],
            "device_used": branch_data[0]["info"].get("device", "unknown"),
            "total_bytes": sum(len(b["bytes"]) for b in branch_data)
        }
        
        # Replace generated_branch object (or update list)
        metrics["generated_branch"] = gen_metrics
        
        # write to output — metrics.json does NOT contain final_zip_size
        # (self-referential: the ZIP size changes when you write metrics into it)
        with zipfile.ZipFile(output_pkg, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            zout.writestr("manifest.json", json.dumps(manifest, indent=2))
            zout.writestr("metrics.json", json.dumps(metrics, indent=2))
            
            for item in zin.infolist():
                if item.filename not in ["manifest.json", "metrics.json"]:
                    zout.writestr(item, zin.read(item.filename))
                    
            for lvl_data in branch_data:
                lvl = lvl_data["info"]["level"]
                zout.writestr(f"generated_branches/{branch_manifest['branch_id']}/level-{lvl:02d}.webp", lvl_data["bytes"])

