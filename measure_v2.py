import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.python.package import read_package
from src.python.renderer import render_image

def eval_layer(pkg, layers, orig_img_np):
    manifest, data, metrics = read_package(pkg, layers=layers)
    W = manifest["encoded_width"]
    H = manifest["encoded_height"]
    data_t = torch.from_numpy(data).cuda()
    
    pos = data_t[:, 0:2]
    scale = data_t[:, 2:4]
    rot = data_t[:, 4]
    color = data_t[:, 5:8]
    opacity = data_t[:, 8]
    
    with torch.no_grad():
        final_img_t = render_image(W, H, pos, scale, rot, color, opacity)
        final_img_np = (final_img_t.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    
    p = psnr(orig_img_np, final_img_np)
    s = ssim(orig_img_np, final_img_np, channel_axis=-1, data_range=255, win_size=3)
    return len(data), p, s

def main():
    orig = np.array(Image.open(r"C:\Users\RUSHABH\Downloads\porche.png").convert('RGB'))
    
    cnt_base, p_base, s_base = eval_layer("porche_10k_progressive_v2.zip", "base", orig)
    print(f"Base Layer ({cnt_base} splats): PSNR={p_base:.4f}, SSIM={s_base:.4f}")
    
    cnt_all, p_all, s_all = eval_layer("porche_10k_progressive_v2.zip", "all", orig)
    print(f"Full Layer ({cnt_all} splats): PSNR={p_all:.4f}, SSIM={s_all:.4f}")

if __name__ == "__main__":
    main()
