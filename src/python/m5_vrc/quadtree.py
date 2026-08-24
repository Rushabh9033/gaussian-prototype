import math
import numpy as np

class QuadtreeNode:
    def __init__(self, z, x, y, bounds):
        self.z = z
        self.x = x
        self.y = y
        self.bounds = bounds # (x_min, y_min, x_max, y_max)
        self.children = []

def build_quadtree_index(logical_w: int, logical_h: int, max_z: int, tile_size: int = 256):
    """
    Builds a quadtree hierarchy covering the logical image scaled up to max_z levels.
    """
    root = QuadtreeNode(0, 0, 0, (0, 0, logical_w, logical_h))
    
    def subdivide(node):
        if node.z >= max_z:
            return
            
        # At next zoom, the logical bounds map to double the dimensions
        nz = node.z + 1
        scale = 2 ** nz
        level_w = logical_w * scale
        level_h = logical_h * scale
        
        # We can explicitly tile this level
        tiles_x = math.ceil(level_w / tile_size)
        tiles_y = math.ceil(level_h / tile_size)
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x_min = tx * tile_size
                y_min = ty * tile_size
                x_max = min((tx + 1) * tile_size, level_w)
                y_max = min((ty + 1) * tile_size, level_h)
                
                child = QuadtreeNode(nz, tx, ty, (x_min, y_min, x_max, y_max))
                node.children.append(child)
                subdivide(child)
                
    subdivide(root)
    return root

def select_visible_tiles(root: QuadtreeNode, target_z: int, viewport: tuple):
    """
    Viewport is (vx_min, vy_min, vx_max, vy_max) at target_z scale.
    """
    visible = []
    
    def traverse(node):
        if node.z == target_z:
            # Check overlap
            b = node.bounds
            if (b[0] < viewport[2] and b[2] > viewport[0] and
                b[1] < viewport[3] and b[3] > viewport[1]):
                visible.append(node)
            return
            
        for child in node.children:
            # We would only traverse children that overlap the viewport projected to their level
            # For simplicity, traverse all until target_z
            traverse(child)
            
    traverse(root)
    return visible

def generate_tiles_for_image(image: np.ndarray, z: int, tile_size: int = 256) -> dict:
    """
    Splits an image into spatial tiles at a given zoom level.
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
