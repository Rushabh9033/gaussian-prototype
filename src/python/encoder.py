import torch
import torch.nn as nn
import torch.optim as optim
import time
from .renderer import compute_covariances, get_active_gaussians, render_tile

class GaussianModel(nn.Module):
    def __init__(self, N, W, H):
        super().__init__()
        # Initialize positions randomly in the image
        self.pos = nn.Parameter(torch.rand(N, 2) * torch.tensor([W, H]))
        # Initialize scales to ~ 1% of image size
        self.log_scale = nn.Parameter(torch.log(torch.ones(N, 2) * (max(W, H) * 0.01)))
        self.rot = nn.Parameter(torch.zeros(N))
        self.color_raw = nn.Parameter(torch.randn(N, 3) * 0.1)
        self.opacity_raw = nn.Parameter(torch.zeros(N))

    def get_params(self):
        scale = torch.exp(self.log_scale)
        color = torch.sigmoid(self.color_raw)
        opacity = torch.sigmoid(self.opacity_raw)
        return self.pos, scale, self.rot, color, opacity

def encode_image(target_img_t, num_gaussians, steps, lr=0.01, tile_size=128, device='cpu', seed=42):
    if seed is not None:
        torch.manual_seed(seed)
        
    H, W = target_img_t.shape[1], target_img_t.shape[2]
    model = GaussianModel(num_gaussians, W, H).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    
    # Pre-sample colors for better initialization
    with torch.no_grad():
        px = torch.clamp(model.pos[:, 0].long(), 0, W - 1)
        py = torch.clamp(model.pos[:, 1].long(), 0, H - 1)
        sampled_colors = target_img_t[:, py, px].T
        # inverse sigmoid approx
        model.color_raw.data = torch.log(sampled_colors / (1.0 - sampled_colors + 1e-6) + 1e-6)
    
    for step in range(steps):
        optimizer.zero_grad()
        total_loss_val = 0.0
        
        # We need the base parameters to slice from
        pos_p = model.pos
        log_scale_p = model.log_scale
        rot_p = model.rot
        color_raw_p = model.color_raw
        opacity_raw_p = model.opacity_raw
        
        # For active gaussian checking, we can use a detached version of scale
        with torch.no_grad():
            scale_detached = torch.exp(log_scale_p)
            
        for ty in range(0, H, tile_size):
            for tx in range(0, W, tile_size):
                cur_W = min(tile_size, W - tx)
                cur_H = min(tile_size, H - ty)
                
                active = get_active_gaussians(tx, ty, cur_W, cur_H, pos_p.detach(), scale_detached)
                active_idx = torch.nonzero(active).squeeze(1)
                
                if len(active_idx) > 0:
                    a_pos = pos_p[active_idx]
                    a_scale = torch.exp(log_scale_p[active_idx])
                    a_rot = rot_p[active_idx]
                    a_color = torch.sigmoid(color_raw_p[active_idx])
                    a_opac = torch.sigmoid(opacity_raw_p[active_idx])
                    
                    A, B, C = compute_covariances(a_scale, a_rot)
                    
                    tile_target = target_img_t[:, ty:ty+cur_H, tx:tx+cur_W]
                    tile_pred = render_tile(
                        tx, ty, cur_W, cur_H,
                        a_pos, A, B, C, a_color, a_opac, device
                    )
                    
                    tile_loss = torch.mean((tile_pred - tile_target) ** 2)
                    weight = (cur_W * cur_H) / (W * H)
                    
                    (tile_loss * weight).backward()
                    total_loss_val += tile_loss.item() * weight
                    
        optimizer.step()
        
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{steps}, MSE: {total_loss_val:.6f}")
            
    encode_time = time.time() - start_time
    return model, encode_time
