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
        
        if version not in ["1.0", "2.0"]:
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
            
            # V2 Progressive
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
            
            if len(out_data) == 0:
                raise ValueError("No layers matched")
            
            data = np.vstack(out_data)
            return manifest, data, metrics
