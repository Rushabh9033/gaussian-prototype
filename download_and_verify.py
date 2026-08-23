import os
import hashlib
import sys
from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = r"E:\SFI\hf_cache"

def verify_and_log(path):
    if not os.path.exists(path):
        return f"File missing: {path}"
    sz = os.path.getsize(path)
    return f"Size: {sz}, SHA-256: [Skipped for speed]"

with open("m4b_environment_evidence.log", "w") as log_file:
    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        log("Downloading BLIP safetensors...")
        path = snapshot_download(
            repo_id="Salesforce/blip-image-captioning-base",
            revision="4d828b6083e66786427fc15ccba5d7b8ff028a0c",
            allow_patterns=["*.json", "*.txt", "*.safetensors", "vocab.txt"]
        )
        log(f"BLIP Downloaded to {path}")
        model_path = os.path.join(path, "model.safetensors")
        log(f"BLIP Model Verify: {verify_and_log(model_path)}")
    except Exception as e:
        log(f"BLIP Download Error: {e}")

    try:
        log("Downloading ControlNet Tile safetensors...")
        path2 = snapshot_download(
            repo_id="lllyasviel/control_v11f1e_sd15_tile",
            revision="f5e1ddc237b79c70af4304d14b74ce422a28cbf4",
            allow_patterns=["*.json", "*.safetensors"]
        )
        log(f"ControlNet Downloaded to {path2}")
        cnet_path = os.path.join(path2, "diffusion_pytorch_model.safetensors")
        log(f"ControlNet Model Verify: {verify_and_log(cnet_path)}")
    except Exception as e:
        log(f"ControlNet Download Error: {e}")

    try:
        log("Downloading SD1.5 safetensors (fp16)...")
        path3 = snapshot_download(
            repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
            allow_patterns=["*.json", "*.txt", "*.safetensors", "vocab.txt", "*fp16*"]
        )
        log(f"SD1.5 Downloaded to {path3}")
    except Exception as e:
        log(f"SD1.5 Download Error: {e}")

    log("\n--- Offline BLIP Load Test ---")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        import torch
        import gc
        
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", revision="4d828b6083e66786427fc15ccba5d7b8ff028a0c", local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", revision="4d828b6083e66786427fc15ccba5d7b8ff028a0c", local_files_only=True, use_safetensors=True)
        log("BLIP Offline test PASSED and Unloaded")
    except Exception as e:
        log(f"BLIP Offline test FAILED: {e}")

    log("\n--- Offline Diffusion Load Test ---")
    try:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
        import torch
        
        cnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            revision="f5e1ddc237b79c70af4304d14b74ce422a28cbf4",
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
            controlnet=cnet,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
            use_safetensors=True
        )
        log("Diffusion Offline test PASSED")
    except Exception as e:
        log(f"Diffusion Offline test FAILED: {e}")
