import numpy as np
from .coordinate_map import generate_output_coordinates
from .anisotropic_sampler import sample_anisotropic

def render_tiled(source_rgb: np.ndarray, out_w: int, out_h: int, 
                 theta_map: np.ndarray, coherence_map: np.ndarray, uncertainty_map: np.ndarray,
                 config: dict, tile_size: int = 512, k_radius: int = 2):
    """
    Renders the enlarged output using overlapping tiles to bound RAM usage.
    Overlap isn't strictly needed for the output assembly since coordinate 
    mapping perfectly aligns, but we process in chunks to prevent large arrays.
    """
    out_rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
    
    for y in range(0, out_h, tile_size):
        for x in range(0, out_w, tile_size):
            th = min(tile_size, out_h - y)
            tw = min(tile_size, out_w - x)
            
            x_in, y_in = generate_output_coordinates(out_w, out_h, source_rgb.shape[1], source_rgb.shape[0], 
                                                     start_x=x, start_y=y, tile_w=tw, tile_h=th)
            
            tile_render = sample_anisotropic(source_rgb, x_in, y_in, 
                                             theta_map, coherence_map, uncertainty_map, 
                                             config, k_radius=k_radius)
            
            out_rgb[y:y+th, x:x+tw] = tile_render
            
    return out_rgb
