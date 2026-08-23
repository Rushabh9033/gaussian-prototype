import argparse
import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.python.package import read_package
from src.python.renderer import render_image
import os

def eval_layer(pkg, layers, orig_img_np, device):
    manifest, data, metrics = read_package(pkg, layers=layers)
    W = manifest["encoded_width"]
    H = manifest["encoded_height"]
    data_t = torch.from_numpy(data).to(device)
    
    pos = data_t[:, 0:2]
    scale = data_t[:, 2:4]
    rot = data_t[:, 4]
    color = data_t[:, 5:8]
    opacity = data_t[:, 8]
    
    with torch.no_grad():
        final_img_t = render_image(W, H, pos, scale, rot, color, opacity)
        final_img_np = (final_img_t.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    
    p = psnr(orig_img_np, final_img_np)
    s = ssim(orig_img_np, final_img_np, channel_axis=-1, data_range=255, win_size=7)
    return len(data), p, s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True)
    parser.add_argument('-p', '--package', required=True)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    orig = np.array(Image.open(args.image).convert('RGB'))
    
    # 1. Base Layer
    cnt_base, p_base, s_base = eval_layer(args.package, "base", orig, device)
    print(f"Decoded Base ({cnt_base}): PSNR = {p_base:.4f} dB, SSIM = {s_base:.4f}")
    
    # 2. Full Layer
    cnt_full, p_full, s_full = eval_layer(args.package, "all", orig, device)
    print(f"Decoded Full ({cnt_full}): PSNR = {p_full:.4f} dB, SSIM = {s_full:.4f}")

    sz = os.path.getsize(args.package)
    print(f"Exact ZIP size: {sz} bytes")

if __name__ == "__main__":
    main()
