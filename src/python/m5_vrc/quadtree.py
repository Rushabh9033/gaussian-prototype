import math
import numpy as np

def generate_tiles(image: np.ndarray, tile_size: int = 256) -> dict:
    """
    Splits an image into 256x256 spatial tiles.
    Returns dict: (x, y) -> np.ndarray
    """
    h, w = image.shape[:2]
    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)
    
    tiles = {}
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y_start = ty * tile_size
            y_end = min((ty + 1) * tile_size, h)
            x_start = tx * tile_size
            x_end = min((tx + 1) * tile_size, w)
            
            tiles[(tx, ty)] = image[y_start:y_end, x_start:x_end]
            
    return tiles
