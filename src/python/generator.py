import torch
from PIL import Image
import numpy as np
import hashlib
import json
import os
import io

def get_sha256(data):
    return hashlib.sha256(data).hexdigest()

class BranchGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-x4-upscaler", device=None, test_mode=False):
        self.model_id = model_id
        self.test_mode = test_mode
        self.pipeline = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.resolved_revision = "main"
        
    def load_model(self):
        if self.test_mode:
            self.resolved_revision = "test-mock"
            return
        from diffusers import StableDiffusionUpscalePipeline
        dtype = torch.float16 if 'cuda' in self.device else torch.float32
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)
        self.resolved_revision = "main"

    def generate_level(self, parent_img_pil, prompt, seed, num_inference_steps=20):
        if self.test_mode:
            img_np = np.array(parent_img_pil.resize((parent_img_pil.width * 4, parent_img_pil.height * 4), Image.Resampling.NEAREST))
            np.random.seed(seed % (2**32))
            noise = np.random.randint(-5, 5, img_np.shape, dtype=np.int16)
            out = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(out)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output = self.pipeline(
            prompt=prompt,
            image=parent_img_pil,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        return output

def extract_crop(img_pil, cx, cy, size):
    # Extracts size x size centered at cx, cy, with deterministic replication padding if out of bounds
    w, h = img_pil.size
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    
    # We use crop which pads with black by default, but requirement says "Handle boundary padding deterministically"
    # To be safe and deterministic, we can just use PIL crop. 
    # But let's do replicate padding just in case.
    img_np = np.array(img_pil)
    out_np = np.zeros((size, size, 3), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            src_y = np.clip(y0 + i, 0, h - 1)
            src_x = np.clip(x0 + j, 0, w - 1)
            out_np[i, j] = img_np[src_y, src_x]
            
    return Image.fromarray(out_np)

def generate_branch(source_img_path, cx, cy, region_size, levels, seed_val, prompt, test_mode=False):
    # region_size 96 means radius 96 -> 192x192 input
    # 192x192 upscaled 4x = 768x768
    # Trim to 512x512
    input_size = region_size * 2 # 192
    if input_size != 192:
        print(f"Warning: region_size {region_size} gives input size {input_size}. Expected 96 for 192 input to yield exactly 512x512.")
        
    core_size = 128
    halo = (input_size - core_size) // 2 # (192 - 128) / 2 = 32
    
    generator = BranchGenerator(test_mode=test_mode)
    generator.load_model()
    
    src_img = Image.open(source_img_path).convert("RGB")
    src_bytes = io.BytesIO()
    src_img.save(src_bytes, format="PNG")
    src_hash = get_sha256(src_bytes.getvalue())
    
    branch_data = []
    
    parent_img = src_img
    parent_cx, parent_cy = cx, cy
    parent_hash = src_hash
    
    branch_id = "branch-001"
    
    cumulative_zoom = 1
    
    for lvl in range(1, levels + 1):
        print(f"Generating level {lvl}...")
        
        # Derive seed: Original source SHA-256, User seed, Branch ID, Level number, Parent content hash
        seed_str = f"{src_hash}_{seed_val}_{branch_id}_{lvl}_{parent_hash}"
        lvl_seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        
        # Extract 192x192
        crop_img = extract_crop(parent_img, parent_cx, parent_cy, input_size)
        
        # Upscale 4x -> 768x768
        upscaled = generator.generate_level(crop_img, prompt, lvl_seed)
        
        # Trim 128px from all sides (32 * 4 = 128)
        trim_px = halo * 4
        final_img = upscaled.crop((trim_px, trim_px, upscaled.width - trim_px, upscaled.height - trim_px))
        
        if final_img.size != (512, 512):
            final_img = final_img.resize((512, 512), Image.Resampling.LANCZOS)
            
        buf = io.BytesIO()
        final_img.save(buf, format="WEBP", lossless=True)
        img_bytes = buf.getvalue()
        img_hash = get_sha256(img_bytes)
        
        cumulative_zoom *= 4
        
        lvl_info = {
            "level": lvl,
            "parent_level": lvl - 1,
            "parent_hash": parent_hash,
            "derived_seed": lvl_seed,
            "parent_crop": [parent_cx - input_size//2, parent_cy - input_size//2, input_size, input_size],
            "root_bbox": [cx - (input_size//2)/cumulative_zoom, cy - (input_size//2)/cumulative_zoom, input_size/cumulative_zoom, input_size/cumulative_zoom],
            "cumulative_zoom": cumulative_zoom,
            "width": final_img.width,
            "height": final_img.height,
            "mime_type": "image/webp",
            "sha256": img_hash,
            "model_id": generator.model_id,
            "model_revision": generator.resolved_revision,
            "prompt": prompt,
            "negative_prompt": "",
            "inference_steps": 20,
            "guidance_config": {},
            "provenance": "generated",
            "creation_software": "ScaleField-Gen-0.1"
        }
        
        branch_data.append({
            "info": lvl_info,
            "bytes": img_bytes
        })
        
        # Setup for next level
        parent_img = final_img
        parent_cx, parent_cy = 256, 256 # Center of 512x512
        parent_hash = img_hash
        
    branch_manifest = {
        "branch_id": branch_id,
        "root_bbox_pixels": [cx - input_size//2, cy - input_size//2, input_size, input_size],
        "root_bbox_normalized": [
            (cx - input_size//2) / src_img.width,
            (cy - input_size//2) / src_img.height,
            input_size / src_img.width,
            input_size / src_img.height
        ],
        "anchor": [cx, cy],
        "level_count": levels,
        "provenance": "synthetic",
        "source_sha256": src_hash,
        "levels": [b["info"] for b in branch_data]
    }
    
    return branch_manifest, branch_data
