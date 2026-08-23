import numpy as np

def quantize_gaussians(pos, scale, rot, color, opacity, W, H, rules=None):
    N = pos.shape[0]
    
    if not (np.isfinite(pos).all() and np.isfinite(rot).all() and np.isfinite(color).all() and np.isfinite(opacity).all()):
        raise ValueError("NaN or Infinity in pos, rot, color, or opacity")
    if not np.isfinite(scale).all():
        raise ValueError("NaN or Infinity in scale")
    if not (scale > 0).all():
        raise ValueError("Scale must be strictly positive")
    
    # Position
    pos_x = np.clip(pos[:, 0] / W, 0.0, 1.0) * 65535
    pos_y = np.clip(pos[:, 1] / H, 0.0, 1.0) * 65535
    
    # Scale
    log_scale = np.log(scale)
    
    if rules is not None:
        min_sx = rules["scale"]["min_log_x"]
        max_sx = rules["scale"]["max_log_x"]
        min_sy = rules["scale"]["min_log_y"]
        max_sy = rules["scale"]["max_log_y"]
    else:
        if N > 0:
            min_sx, max_sx = float(np.min(log_scale[:, 0])), float(np.max(log_scale[:, 0]))
            min_sy, max_sy = float(np.min(log_scale[:, 1])), float(np.max(log_scale[:, 1]))
        else:
            min_sx, max_sx, min_sy, max_sy = 0.0, 1.0, 0.0, 1.0
        
        if max_sx == min_sx: max_sx += 1e-5
        if max_sy == min_sy: max_sy += 1e-5
    
    sc_x = np.clip((log_scale[:, 0] - min_sx) / (max_sx - min_sx), 0.0, 1.0) * 65535
    sc_y = np.clip((log_scale[:, 1] - min_sy) / (max_sy - min_sy), 0.0, 1.0) * 65535
    
    # Rotation
    rot_norm = rot % (2 * np.pi)
    r_q = (rot_norm / (2 * np.pi)) * 65535
    
    # Colors & Opacity
    c_r = np.clip(color[:, 0], 0.0, 1.0) * 255
    c_g = np.clip(color[:, 1], 0.0, 1.0) * 255
    c_b = np.clip(color[:, 2], 0.0, 1.0) * 255
    
    opac = np.clip(opacity, 0.0, 1.0) * 255
    
    # Pack into structured array
    # '<H' is little-endian uint16, '<B' is little-endian uint8
    dtype = np.dtype([
        ('x', '<u2'), ('y', '<u2'),
        ('sx', '<u2'), ('sy', '<u2'),
        ('rot', '<u2'),
        ('r', '<u1'), ('g', '<u1'), ('b', '<u1'), ('a', '<u1')
    ])
    
    data = np.zeros(N, dtype=dtype)
    data['x'] = pos_x.astype(np.uint16)
    data['y'] = pos_y.astype(np.uint16)
    data['sx'] = sc_x.astype(np.uint16)
    data['sy'] = sc_y.astype(np.uint16)
    data['rot'] = r_q.astype(np.uint16)
    data['r'] = c_r.astype(np.uint8)
    data['g'] = c_g.astype(np.uint8)
    data['b'] = c_b.astype(np.uint8)
    data['a'] = opac.astype(np.uint8)
    
    quant_rules = {
        "position": {"x_max": float(W), "y_max": float(H)},
        "scale": {
            "min_log_x": min_sx, "max_log_x": max_sx,
            "min_log_y": min_sy, "max_log_y": max_sy
        },
        "rotation": {"range": [0.0, float(2 * np.pi)]},
        "color": {"range": [0.0, 1.0]},
        "opacity": {"range": [0.0, 1.0]}
    }
    
    return data.tobytes(), quant_rules

def dequantize_gaussians(bin_data, rules):
    dtype = np.dtype([
        ('x', '<u2'), ('y', '<u2'),
        ('sx', '<u2'), ('sy', '<u2'),
        ('rot', '<u2'),
        ('r', '<u1'), ('g', '<u1'), ('b', '<u1'), ('a', '<u1')
    ])
    
    data = np.frombuffer(bin_data, dtype=dtype)
    N = len(data)
    
    W = rules["position"]["x_max"]
    H = rules["position"]["y_max"]
    
    pos = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] = (data['x'].astype(np.float32) / 65535.0) * W
    pos[:, 1] = (data['y'].astype(np.float32) / 65535.0) * H
    
    min_sx, max_sx = rules["scale"]["min_log_x"], rules["scale"]["max_log_x"]
    min_sy, max_sy = rules["scale"]["min_log_y"], rules["scale"]["max_log_y"]
    
    log_sx = (data['sx'].astype(np.float32) / 65535.0) * (max_sx - min_sx) + min_sx
    log_sy = (data['sy'].astype(np.float32) / 65535.0) * (max_sy - min_sy) + min_sy
    
    scale = np.zeros((N, 2), dtype=np.float32)
    scale[:, 0] = np.exp(log_sx)
    scale[:, 1] = np.exp(log_sy)
    
    rot = (data['rot'].astype(np.float32) / 65535.0) * float(2 * np.pi)
    
    color = np.zeros((N, 3), dtype=np.float32)
    color[:, 0] = data['r'].astype(np.float32) / 255.0
    color[:, 1] = data['g'].astype(np.float32) / 255.0
    color[:, 2] = data['b'].astype(np.float32) / 255.0
    
    opacity = data['a'].astype(np.float32) / 255.0
    
    out = np.zeros((N, 9), dtype=np.float32)
    out[:, 0:2] = pos
    out[:, 2:4] = scale
    out[:, 4] = rot
    out[:, 5:8] = color
    out[:, 8] = opacity
    
    return out
