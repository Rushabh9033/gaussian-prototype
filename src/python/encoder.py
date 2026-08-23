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

def compute_loss(pred, target, strategy='random'):
    if strategy == 'random':
        return torch.mean((pred - target) ** 2)
    else:
        # L1 loss
        l1 = torch.abs(pred - target).mean()
        # Edge preservation loss (Sobel-like)
        dy_pred = pred[:, 1:, :] - pred[:, :-1, :]
        dy_tgt = target[:, 1:, :] - target[:, :-1, :]
        dx_pred = pred[:, :, 1:] - pred[:, :, :-1]
        dx_tgt = target[:, :, 1:] - target[:, :, :-1]
        grad_loss = torch.abs(dy_pred - dy_tgt).mean() + torch.abs(dx_pred - dx_tgt).mean()
        return l1 + 0.1 * grad_loss

def encode_image(target_img_t, num_gaussians, steps, lr=0.01, tile_size=128, device='cpu', seed=42, strategy='random'):
    if seed is not None:
        torch.manual_seed(seed)
        
    H, W = target_img_t.shape[1], target_img_t.shape[2]
    model = GaussianModel(num_gaussians, W, H).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    
    if strategy == 'adaptive':
        active_count = int(0.6 * num_gaussians)
        with torch.no_grad():
            gray = 0.299 * target_img_t[0] + 0.587 * target_img_t[1] + 0.114 * target_img_t[2]
            dy = torch.zeros_like(gray)
            dx = torch.zeros_like(gray)
            dy[:-1, :] = torch.abs(gray[1:, :] - gray[:-1, :])
            dx[:, :-1] = torch.abs(gray[:, 1:] - gray[:, :-1])
            edges = dy + dx
            edges = torch.nn.functional.avg_pool2d(edges.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
            density = edges + 0.1 * edges.max() # uniform floor
            density = density / density.sum()
            
            flat_indices = torch.multinomial(density.flatten(), active_count, replacement=True)
            y_coords = (flat_indices // W).float() + torch.rand(active_count, device=device)
            x_coords = (flat_indices % W).float() + torch.rand(active_count, device=device)
            
            model.pos.data[:active_count, 0] = x_coords
            model.pos.data[:active_count, 1] = y_coords
            model.log_scale.data[:active_count, :] = torch.log(torch.ones(active_count, 2, device=device) * (max(W, H) * 0.005))
            
            px = torch.clamp(model.pos[:active_count, 0].long(), 0, W - 1)
            py = torch.clamp(model.pos[:active_count, 1].long(), 0, H - 1)
            sampled_colors = target_img_t[:, py, px].T
            model.color_raw.data[:active_count] = torch.log(sampled_colors / (1.0 - sampled_colors + 1e-6) + 1e-6)
            model.opacity_raw.data[:active_count] = 0.0
    else:
        active_count = num_gaussians
        with torch.no_grad():
            px = torch.clamp(model.pos[:, 0].long(), 0, W - 1)
            py = torch.clamp(model.pos[:, 1].long(), 0, H - 1)
            sampled_colors = target_img_t[:, py, px].T
            model.color_raw.data = torch.log(sampled_colors / (1.0 - sampled_colors + 1e-6) + 1e-6)
    
    for step in range(steps):
        if strategy == 'adaptive' and step == steps // 2:
            print(f"--- Stage 2: Adding residual-guided Gaussians at step {step} ---")
            with torch.no_grad():
                full_pred = torch.zeros_like(target_img_t)
                pos_p = model.pos[:active_count]
                scale_p = torch.exp(model.log_scale[:active_count])
                rot_p = model.rot[:active_count]
                color_p = torch.sigmoid(model.color_raw[:active_count])
                opac_p = torch.sigmoid(model.opacity_raw[:active_count])
                A, B, C = compute_covariances(scale_p, rot_p)
                for ty in range(0, H, tile_size):
                    for tx in range(0, W, tile_size):
                        cur_W = min(tile_size, W - tx)
                        cur_H = min(tile_size, H - ty)
                        act = get_active_gaussians(tx, ty, cur_W, cur_H, pos_p, scale_p)
                        act_idx = torch.nonzero(act).squeeze(1)
                        if len(act_idx) > 0:
                            tp = render_tile(tx, ty, cur_W, cur_H, pos_p[act_idx], A[act_idx], B[act_idx], C[act_idx], color_p[act_idx], opac_p[act_idx], device)
                            full_pred[:, ty:ty+cur_H, tx:tx+cur_W] = tp
                            
                residual = torch.abs(full_pred - target_img_t).mean(dim=0)
                density = residual + 0.05 * residual.max()
                density = density / density.sum()
                
                N2 = num_gaussians - active_count
                flat_indices = torch.multinomial(density.flatten(), N2, replacement=True)
                y_coords = (flat_indices // W).float() + torch.rand(N2, device=device)
                x_coords = (flat_indices % W).float() + torch.rand(N2, device=device)
                
                model.pos.data[active_count:, 0] = x_coords
                model.pos.data[active_count:, 1] = y_coords
                model.log_scale.data[active_count:, :] = torch.log(torch.ones(N2, 2, device=device) * (max(W, H) * 0.003))
                
                px = torch.clamp(model.pos[active_count:, 0].long(), 0, W - 1)
                py = torch.clamp(model.pos[active_count:, 1].long(), 0, H - 1)
                sampled_colors = target_img_t[:, py, px].T
                model.color_raw.data[active_count:] = torch.log(sampled_colors / (1.0 - sampled_colors + 1e-6) + 1e-6)
                model.opacity_raw.data[active_count:] = 0.0
                
            active_count = num_gaussians
            
        optimizer.zero_grad()
        total_loss_val = 0.0
        
        pos_p = model.pos[:active_count]
        log_scale_p = model.log_scale[:active_count]
        rot_p = model.rot[:active_count]
        color_raw_p = model.color_raw[:active_count]
        opacity_raw_p = model.opacity_raw[:active_count]
        
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
                    
                    tile_loss = compute_loss(tile_pred, tile_target, strategy)
                    weight = (cur_W * cur_H) / (W * H)
                    
                    (tile_loss * weight).backward()
                    total_loss_val += tile_loss.item() * weight
                    
        optimizer.step()
        
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{steps}, Loss: {total_loss_val:.6f}")
            
    encode_time = time.time() - start_time
    return model, encode_time
