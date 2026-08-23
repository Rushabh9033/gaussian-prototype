import torch

def compute_covariances(scale, rot):
    # scale: N x 2, rot: N
    cos_r = torch.cos(rot)
    sin_r = torch.sin(rot)
    
    sx_inv2 = 1.0 / (scale[:, 0] ** 2 + 1e-8)
    sy_inv2 = 1.0 / (scale[:, 1] ** 2 + 1e-8)
    
    A = cos_r**2 * sx_inv2 + sin_r**2 * sy_inv2
    B = cos_r * sin_r * (sx_inv2 - sy_inv2)
    C = sin_r**2 * sx_inv2 + cos_r**2 * sy_inv2
    return A, B, C

def get_active_gaussians(tx, ty, cur_W, cur_H, pos, scale):
    tile_cx = tx + cur_W / 2.0
    tile_cy = ty + cur_H / 2.0
    
    max_scale = torch.max(scale, dim=1)[0]
    # 3 sigma radius
    radius = 3.0 * max_scale
    
    dist_x = torch.abs(pos[:, 0] - tile_cx) - (cur_W / 2.0) - radius
    dist_y = torch.abs(pos[:, 1] - tile_cy) - (cur_H / 2.0) - radius
    
    active = (dist_x < 0) & (dist_y < 0)
    return active

def render_tile(tx, ty, cur_W, cur_H, pos, A, B, C, color, opacity, device):
    N = pos.shape[0]
    if N == 0:
        return torch.zeros(3, cur_H, cur_W, device=device)
        
    y_grid = torch.arange(ty, ty + cur_H, device=device, dtype=torch.float32) + 0.5
    x_grid = torch.arange(tx, tx + cur_W, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # N x cur_H x cur_W
    dx = gx.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
    dy = gy.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
    
    power = -0.5 * (A.view(-1, 1, 1) * dx**2 + 2 * B.view(-1, 1, 1) * dx * dy + C.view(-1, 1, 1) * dy**2)
    
    alpha = opacity.view(-1, 1, 1) * torch.exp(power)
    # Clamp to avoid 1.0 which makes 1-alpha 0 and zeroes out gradients for anything behind
    alpha = torch.clamp(alpha, 0.0, 0.999)
    
    # alpha compositing front-to-back
    ones = torch.ones(1, cur_H, cur_W, device=device)
    if N > 1:
        T = torch.cat([ones, torch.cumprod(1.0 - alpha[:-1], dim=0)], dim=0)
    else:
        T = ones
        
    weight = alpha * T
    tile_color = torch.sum(weight.unsqueeze(1) * color.view(N, 3, 1, 1), dim=0)
    
    return tile_color

def render_image(W, H, pos, scale, rot, color, opacity, tile_size=128):
    device = pos.device
    A, B, C = compute_covariances(scale, rot)
    
    full_image = torch.zeros(3, H, W, device=device)
    
    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            cur_W = min(tile_size, W - tx)
            cur_H = min(tile_size, H - ty)
            
            active = get_active_gaussians(tx, ty, cur_W, cur_H, pos, scale)
            active_idx = torch.nonzero(active).squeeze(1)
            
            if len(active_idx) > 0:
                tile_color = render_tile(
                    tx, ty, cur_W, cur_H,
                    pos[active_idx],
                    A[active_idx], B[active_idx], C[active_idx],
                    color[active_idx], opacity[active_idx],
                    device
                )
                full_image[:, ty:ty+cur_H, tx:tx+cur_W] = tile_color
                
    return full_image
