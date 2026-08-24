from PIL import Image
from typing import Tuple

def sample_and_resize(img_path: str, box_pixels: Tuple[int, int, int, int], target_size: int = 512) -> Image.Image:
    """
    Extract the exact source crop corresponding to each zoom level.
    Resize that crop to 512x512 with a documented high-quality resampler (LANCZOS).
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        
        # Ensure we don't go out of bounds, though coordinates module should handle it
        left, top, right, bottom = box_pixels
        left = max(0, left)
        top = max(0, top)
        right = min(img.width, right)
        bottom = min(img.height, bottom)
        
        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
        return resized
