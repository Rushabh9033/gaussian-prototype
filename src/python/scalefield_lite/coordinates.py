import json
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ZoomLevel:
    level: int
    zoom_factor: float
    box_normalized: Tuple[float, float, float, float] # cx, cy, w, h
    box_pixels: Tuple[int, int, int, int] # left, top, right, bottom

def calculate_zoom_levels(img_width: int, img_height: int, target_cx: float, target_cy: float, zoom_factors: List[float]) -> List[ZoomLevel]:
    levels = []
    for i, zoom in enumerate(zoom_factors):
        # The width and height in normalized coordinates
        w_norm = 1.0 / zoom
        h_norm = 1.0 / zoom
        
        # Calculate bounding box
        left_norm = target_cx - w_norm / 2.0
        top_norm = target_cy - h_norm / 2.0
        right_norm = target_cx + w_norm / 2.0
        bottom_norm = target_cy + h_norm / 2.0
        
        # Clip to [0, 1] - note that at 1x, cx/cy must be 0.5 to fit exactly. 
        # Actually, if zoom=1, we just take the whole image.
        if zoom == 1.0:
            left_norm, top_norm, right_norm, bottom_norm = 0.0, 0.0, 1.0, 1.0
        
        # Pixel coordinates
        left_px = int(round(left_norm * img_width))
        top_px = int(round(top_norm * img_height))
        right_px = int(round(right_norm * img_width))
        bottom_px = int(round(bottom_norm * img_height))
        
        levels.append(ZoomLevel(
            level=i + 1,
            zoom_factor=zoom,
            box_normalized=(target_cx, target_cy, w_norm, h_norm),
            box_pixels=(left_px, top_px, right_px, bottom_px)
        ))
    return levels
