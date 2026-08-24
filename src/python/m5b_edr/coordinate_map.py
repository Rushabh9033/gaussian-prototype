import numpy as np

def generate_output_coordinates(out_w: int, out_h: int, src_w: int, src_h: int, 
                                start_x: int = 0, start_y: int = 0, 
                                tile_w: int = None, tile_h: int = None):
    """
    Generates a grid of output pixel coordinates mapped to the source continuous space.
    Uses pixel-center correct mapping: (x + 0.5) / scale - 0.5.
    If tile dimensions are provided, it only generates coordinates for that tile.
    """
    if tile_w is None: tile_w = out_w
    if tile_h is None: tile_h = out_h
    
    scale_x = out_w / float(src_w)
    scale_y = out_h / float(src_h)
    
    x_out = np.arange(start_x, start_x + tile_w, dtype=np.float32)
    y_out = np.arange(start_y, start_y + tile_h, dtype=np.float32)
    
    xv, yv = np.meshgrid(x_out, y_out)
    
    x_in = (xv + 0.5) / scale_x - 0.5
    y_in = (yv + 0.5) / scale_y - 0.5
    
    return x_in, y_in
