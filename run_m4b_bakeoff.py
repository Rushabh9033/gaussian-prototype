import os
os.environ["HF_HOME"] = r"E:\SFI\hf_cache"
os.makedirs(r"E:\SFI\hf_cache", exist_ok=True)

import gc
import json
import time
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

def get_gradient_energy(img_array):
    gx = np.gradient(img_array, axis=1)
    gy = np.gradient(img_array, axis=0)
    return float(np.mean(gx**2 + gy**2))

def get_laplacian_var(img_array):
    import cv2
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def get_entropy(img_array):
    import cv2
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def calculate_metrics(child_img, parent_img):
    import lpips
    from torchvision.transforms import functional as F
    
    child_reduced = child_img.resize((128, 128), Image.BICUBIC)
    parent_crop = parent_img.crop((192, 192, 320, 320))
    
    import cv2
    c_arr = np.array(child_reduced)
    p_arr = np.array(parent_crop)
    gray_c = cv2.cvtColor(c_arr, cv2.COLOR_RGB2GRAY)
    gray_p = cv2.cvtColor(p_arr, cv2.COLOR_RGB2GRAY)
    from skimage.metrics import structural_similarity as ssim
    ssim_val = ssim(gray_p, gray_c, data_range=255)
    
    edge_c = cv2.Canny(gray_c, 100, 200)
    edge_p = cv2.Canny(gray_p, 100, 200)
    edge_cons = float(np.mean(edge_c == edge_p))
    
    loss_fn = lpips.LPIPS(net='alex')
    t_c = F.to_tensor(child_reduced).unsqueeze(0) * 2 - 1
    t_p = F.to_tensor(parent_crop).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        lpips_val = float(loss_fn(t_p, t_c).item())
        
    full_c_arr = np.array(child_img)
    return {
        "ssim_to_parent": float(ssim_val),
        "lpips_to_parent": lpips_val,
        "edge_consistency": edge_cons,
        "gradient_energy": get_gradient_energy(full_c_arr),
        "laplacian_variance": get_laplacian_var(full_c_arr),
        "entropy": get_entropy(full_c_arr)
    }

def run_experiment():
    out_dir = "m4b_results/candidate_c"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading VLM...")
    vlm_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    orig_img = Image.open("porche.png").convert("RGB")
    l1_crop = orig_img.resize((512, 512), resample=Image.BICUBIC, box=(181, 86, 309, 214))
    
    inputs = vlm_proc(l1_crop, "a close up of", return_tensors="pt")
    out = vlm_model.generate(**inputs)
    caption = vlm_proc.decode(out[0], skip_special_tokens=True)
    print(f"Caption generated: {caption}")
    
    del vlm_proc, vlm_model, inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Loading Diffusion Pipeline...")
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    ctrl_id = "lllyasviel/control_v11f1e_sd15_tile"
    
    try:
        controlnet = ControlNetModel.from_pretrained(ctrl_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Failed to load Tile ControlNet. Fallback to Canny. Error: {e}")
        ctrl_id = "lllyasviel/control_v11p_sd15_canny"
        controlnet = ControlNetModel.from_pretrained(ctrl_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch.float16
        )
        
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print("Model CPU offload failed, trying sequential:", e)
        pipe.enable_sequential_cpu_offload()
        
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    cx, cy = 245.0, 150.0
    sizes = [128.0, 32.0, 8.0, 2.0, 0.5]
    levels = []
    metrics = {}
    prompts = {}
    
    parent_img = None
    generator = torch.Generator(device="cpu").manual_seed(42)
    
    peak_vram = 0
    start_total = time.time()
    
    for i, sz in enumerate(sizes):
        lvl = i + 1
        print(f"Generating level {lvl} (size {sz})...")
        t0 = time.time()
        
        box = (cx - sz/2, cy - sz/2, cx + sz/2, cy + sz/2)
        ctrl_img = orig_img.resize((512, 512), resample=Image.BICUBIC, box=box)
        
        if ctrl_id == "lllyasviel/control_v11p_sd15_canny":
            import cv2
            ctrl_arr = np.array(ctrl_img)
            edges = cv2.Canny(ctrl_arr, 100, 200)
            ctrl_img = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
            
        init_img = ctrl_img if parent_img is None else parent_img.resize((512, 512), resample=Image.BICUBIC, box=(192, 192, 320, 320))
        
        prompt = f"{caption}, ultra-detailed, 8k resolution, macro photography, microscopic detail, scale level {lvl}"
        prompts[f"level_{lvl}"] = prompt
        
        torch.cuda.reset_peak_memory_stats()
        
        try:
            out_img = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, pixelated, smooth, out of focus",
                image=init_img,
                control_image=ctrl_img,
                num_inference_steps=20,
                generator=generator,
                strength=0.8,
                controlnet_conditioning_scale=1.0
            ).images[0]
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print("OOM detected. Retrying with sequential offload...")
                pipe.enable_sequential_cpu_offload()
                out_img = pipe(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, pixelated",
                    image=init_img,
                    control_image=ctrl_img,
                    num_inference_steps=20,
                    generator=generator,
                    strength=0.8,
                    controlnet_conditioning_scale=1.0
                ).images[0]
            else:
                raise e
                
        t1 = time.time()
        vram = torch.cuda.max_memory_allocated() / 1e9
        peak_vram = max(peak_vram, vram)
        
        out_path = os.path.join(out_dir, f"level_{lvl}.png")
        out_img.save(out_path)
        
        if parent_img is not None:
            m = calculate_metrics(out_img, parent_img)
            m["runtime"] = t1 - t0
            m["file_size"] = os.path.getsize(out_path)
            metrics[f"level_{lvl}"] = m
        else:
            full_c = np.array(out_img)
            metrics[f"level_{lvl}"] = {
                "runtime": t1 - t0,
                "file_size": os.path.getsize(out_path),
                "gradient_energy": get_gradient_energy(full_c),
                "laplacian_variance": get_laplacian_var(full_c),
                "entropy": get_entropy(full_c)
            }
            
        levels.append(out_img)
        parent_img = out_img
        
    print("Creating contact sheet...")
    contact = Image.new('RGB', (512 * 5, 512))
    for i, img in enumerate(levels):
        contact.paste(img, (i * 512, 0))
    contact.save(os.path.join(out_dir, "contact_sheet.png"))
    
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
        
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    prov = {
        "model_id": model_id,
        "controlnet_id": ctrl_id,
        "vlm_id": "Salesforce/blip-image-captioning-base",
        "peak_vram_gb": peak_vram,
        "total_runtime": time.time() - start_total
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        
    print("Done! Metrics:", json.dumps(metrics, indent=2))
    
    status = {
      "candidates": {
        "C": {"name": "Direct-from-Root Structure-Conditioned", "status": "NEEDS_USER_VISUAL_CHECK", "reason": "Images successfully generated"}
      },
      "overall_status": "NEEDS_USER_VISUAL_CHECK"
    }
    with open("m4b_results.json", "w") as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    run_experiment()
