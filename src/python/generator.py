import torch
from PIL import Image
import numpy as np
import hashlib
import json
import os
import io
import time

def get_sha256(data):
    return hashlib.sha256(data).hexdigest()

class BranchGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-x4-upscaler", model_revision="main", model_path="", device=None, test_mode=False):
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_path = model_path
        self.test_mode = test_mode
        self.pipeline = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.resolved_revision = "main"
        
    def load_model(self):
        if self.test_mode:
            self.resolved_revision = "test-mock"
            return
            
        import sys
        is_test = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.argv[0]
        if self.model_revision == "test-mock" and not is_test:
            raise ValueError("Cannot load test-mock in production.")
            
        from diffusers import StableDiffusionUpscalePipeline
        dtype = torch.float16 if 'cuda' in self.device else torch.float32
        
        # Load local or remote
        try:
            if self.model_path:
                self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                    self.model_path, torch_dtype=dtype, local_files_only=True
                ).to(self.device)
                self.resolved_revision = "local"
            else:
                from huggingface_hub import model_info
                info = model_info(self.model_id, revision=self.model_revision)
                self.resolved_revision = info.sha
                
                self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                    self.model_id, revision=self.model_revision, torch_dtype=dtype
                ).to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_level(self, parent_img_pil, prompt, negative_prompt, seed, num_inference_steps=20):
        if self.test_mode:
            # 160x160 -> 640x640 mock
            img_np = np.array(parent_img_pil.resize((parent_img_pil.width * 4, parent_img_pil.height * 4), Image.Resampling.NEAREST))
            np.random.seed(seed % (2**32))
            noise = np.random.randint(-5, 5, img_np.shape, dtype=np.int16)
            out = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            time.sleep(0.1) # fake delay
            return Image.fromarray(out)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=parent_img_pil,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        return output

def extract_crop(img_pil, cx, cy, size):
    w, h = img_pil.size
    half = size // 2
    x0, y0 = cx - half, cy - half
    
    img_np = np.array(img_pil)
    out_np = np.zeros((size, size, 3), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            src_y = np.clip(y0 + i, 0, h - 1)
            src_x = np.clip(x0 + j, 0, w - 1)
            out_np[i, j] = img_np[src_y, src_x]
            
    return Image.fromarray(out_np)

def generate_branch(source_img_path, cx, cy, region_size, levels, seed_val, prompt, negative_prompt="", model_id="stabilityai/stable-diffusion-x4-upscaler", model_revision="main", model_path="", device=None, inference_steps=20, test_mode=False):
    if region_size != 128:
        print(f"Warning: Region size should be 128 as per spec. Got {region_size}.")
        
    core_size = region_size
    halo = 16
    input_size = core_size + halo * 2 # 160
    
    generator = BranchGenerator(model_id=model_id, model_revision=model_revision, model_path=model_path, device=device, test_mode=test_mode)
    generator.load_model()
    
    src_img = Image.open(source_img_path).convert("RGB")
    
    # Read actual file bytes for hash
    with open(source_img_path, "rb") as f:
        src_bytes_raw = f.read()
    src_hash = get_sha256(src_bytes_raw)
    
    # Bounds check
    if cx - core_size // 2 < 0 or cy - core_size // 2 < 0 or cx + core_size // 2 > src_img.width or cy + core_size // 2 > src_img.height:
        raise ValueError("Invalid coordinates: Bounding box falls outside source image.")
    
    branch_data = []
    
    parent_img = src_img
    parent_cx, parent_cy = cx, cy
    parent_hash = src_hash
    
    branch_id = "branch-001"
    
    cumulative_zoom = 1
    
    for lvl in range(1, levels + 1):
        print(f"Generating level {lvl}...")
        
        t0 = time.time()
        
        seed_str = f"{src_hash}_{seed_val}_{branch_id}_{lvl}_{parent_hash}"
        lvl_seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        
        crop_img = extract_crop(parent_img, parent_cx, parent_cy, input_size)
        
        upscaled = generator.generate_level(crop_img, prompt, negative_prompt, lvl_seed, inference_steps)
        
        trim_px = halo * 4 # 64
        final_img = upscaled.crop((trim_px, trim_px, upscaled.width - trim_px, upscaled.height - trim_px))
        
        if final_img.size != (512, 512):
            final_img = final_img.resize((512, 512), Image.Resampling.LANCZOS)
            
        buf = io.BytesIO()
        final_img.save(buf, format="WEBP", lossless=True)
        img_bytes = buf.getvalue()
        img_hash = get_sha256(img_bytes)
        
        t1 = time.time()
        
        import skimage.metrics
        
        # Calculate parent consistency
        # parent core is the central 128x128 of the 160x160 crop_img
        parent_core = crop_img.crop((halo, halo, halo + core_size, halo + core_size))
        child_downscaled = final_img.resize((core_size, core_size), Image.Resampling.LANCZOS)
        
        pc_np = np.array(parent_core)
        cd_np = np.array(child_downscaled)
        
        psnr_val = skimage.metrics.peak_signal_noise_ratio(pc_np, cd_np)
        ssim_val = skimage.metrics.structural_similarity(pc_np, cd_np, data_range=255, channel_axis=2, win_size=7)
        
        cumulative_zoom *= 4
        
        lvl_info = {
            "level": lvl,
            "parent_level": lvl - 1,
            "parent_hash": parent_hash,
            "derived_seed": lvl_seed,
            "parent_crop": [parent_cx - input_size//2, parent_cy - input_size//2, input_size, input_size],
            "root_bbox": [cx - (core_size//2)/cumulative_zoom, cy - (core_size//2)/cumulative_zoom, core_size/cumulative_zoom, core_size/cumulative_zoom],
            "cumulative_zoom": cumulative_zoom,
            "width": final_img.width,
            "height": final_img.height,
            "mime_type": "image/webp",
            "sha256": img_hash,
            "model_id": generator.model_id,
            "model_revision": generator.resolved_revision,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "inference_steps": inference_steps,
            "guidance_config": {},
            "provenance": "generated",
            "creation_software": "ScaleField-Gen-0.1",
            "generation_time_seconds": t1 - t0,
            "parent_consistency_psnr": float(psnr_val),
            "parent_consistency_ssim": float(ssim_val),
            "byte_size": len(img_bytes)
        }
        
        branch_data.append({
            "info": lvl_info,
            "bytes": img_bytes
        })
        
        parent_img = final_img
        parent_cx, parent_cy = 256, 256 # Center of 512x512
        parent_hash = img_hash
        
    branch_manifest = {
        "branch_id": branch_id,
        "root_bbox_pixels": [cx - core_size//2, cy - core_size//2, core_size, core_size],
        "root_bbox_normalized": [
            (cx - core_size//2) / src_img.width,
            (cy - core_size//2) / src_img.height,
            core_size / src_img.width,
            core_size / src_img.height
        ],
        "anchor": [cx, cy],
        "level_count": levels,
        "provenance": "synthetic",
        "source_sha256": src_hash,
        "levels": [b["info"] for b in branch_data]
    }
    
    return branch_manifest, branch_data
