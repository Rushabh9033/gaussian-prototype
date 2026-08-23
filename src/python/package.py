import json
import struct
import zipfile
import numpy as np
import io
from PIL import Image

def create_package(zip_path, pos, scale, rot, color, opacity, W, H, metrics, preview_img=None, seed=42):
    # Prepare binary data
    N = pos.shape[0]
    
    # Pack as float32
    # array of shape (N, 9)
    # [x, y, scale_x, scale_y, rot, r, g, b, a]
    data = np.zeros((N, 9), dtype=np.float32)
    data[:, 0:2] = pos
    data[:, 2:4] = scale
    data[:, 4] = rot
    data[:, 5:8] = color
    data[:, 8] = opacity
    
    splats_bin = data.tobytes() # little-endian by default on x86/ARM
    
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

def read_package(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        manifest = json.loads(zf.read("manifest.json"))
        metrics = json.loads(zf.read("metrics.json"))
        
        splats_bin = zf.read("splats.bin")
        data = np.frombuffer(splats_bin, dtype=np.float32).reshape(-1, 9)
        
        return manifest, data, metrics
