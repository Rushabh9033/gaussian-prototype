import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

class ScaleFieldGenerator:
    def __init__(self):
        self.model_id = "stabilityai/sd-turbo"
        self.revision = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_id} (revision: {self.revision}) on {self.device}...")
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.pipeline.to(self.device)
        
        # Optimize memory if needed, but RTX 4060 has 8.59GB, SD-Turbo fits easily
        # self.pipeline.enable_attention_slicing() # only if required

    def generate(self, init_image: Image.Image, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # SD-Turbo operates best with 1-4 steps. We'll use 2 for img2img.
        # guidance_scale is typically 0.0 for SD-Turbo.
        result = self.pipeline(
            prompt=prompt,
            image=init_image,
            num_inference_steps=2,
            strength=0.5, # moderate variation
            guidance_scale=0.0,
            generator=generator
        ).images[0]
        
        return result
