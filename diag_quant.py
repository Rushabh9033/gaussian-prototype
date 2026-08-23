import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.python.package import read_package
from src.python.renderer import render_image
from src.python.quantize import quantize_gaussians, dequantize_gaussians

def test_quant():
    orig = np.array(Image.open(r"C:\Users\RUSHABH\Downloads\porche.png").convert('RGB'))
    manifest, data, metrics = read_package("porche_10k_progressive_v2.zip", layers="all")
    # this data is already quantized/dequantized.
    # We need the original unquantized data. Wait, how to get it?
    # I'll just re-encode for 50 steps and compare.
    pass

def test_quant2():
    from src.python.encoder import encode_image
    orig = np.array(Image.open(r"C:\Users\RUSHABH\Downloads\porche.png").convert('RGB'))
    img_t = torch.from_numpy(orig).float().permute(2, 0, 1) / 255.0
    model, _ = encode_image(img_t.cuda(), 10000, 150, strategy='adaptive', seed=42, device='cuda')
    pos, scale, rot, color, opacity = model.get_params()
    pos = pos.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rot = rot.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    W, H = 360, 220
    
    # render original
    with torch.no_grad():
        final = render_image(W, H, torch.from_numpy(pos).cuda(), torch.from_numpy(scale).cuda(), torch.from_numpy(rot).cuda(), torch.from_numpy(color).cuda(), torch.from_numpy(opacity).cuda())
        final = (final.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    p_orig = psnr(orig, final)
    print(f"Original PSNR: {p_orig}")
    
    bin_data, rules = quantize_gaussians(pos, scale, rot, color, opacity, W, H)
    dequant = dequantize_gaussians(bin_data, rules)
    pos_dq = dequant[:, 0:2]
    scale_dq = dequant[:, 2:4]
    rot_dq = dequant[:, 4]
    color_dq = dequant[:, 5:8]
    opacity_dq = dequant[:, 8]
    
    with torch.no_grad():
        final_dq = render_image(W, H, torch.from_numpy(pos_dq).cuda(), torch.from_numpy(scale_dq).cuda(), torch.from_numpy(rot_dq).cuda(), torch.from_numpy(color_dq).cuda(), torch.from_numpy(opacity_dq).cuda())
        final_dq = (final_dq.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    p_dq = psnr(orig, final_dq)
    print(f"Total Dequantized PSNR: {p_dq}")
    
    # One by one
    for name, dq_arr, og_arr in [("pos", pos_dq, pos), ("scale", scale_dq, scale), 
                                 ("rot", rot_dq, rot), ("color", color_dq, color), 
                                 ("opacity", opacity_dq, opacity)]:
        
        tmp_pos = pos_dq if name == "pos" else pos
        tmp_scale = scale_dq if name == "scale" else scale
        tmp_rot = rot_dq if name == "rot" else rot
        tmp_color = color_dq if name == "color" else color
        tmp_opac = opacity_dq if name == "opacity" else opacity
        
        with torch.no_grad():
            final_tmp = render_image(W, H, torch.from_numpy(tmp_pos).cuda(), torch.from_numpy(tmp_scale).cuda(), torch.from_numpy(tmp_rot).cuda(), torch.from_numpy(tmp_color).cuda(), torch.from_numpy(tmp_opac).cuda())
            final_tmp = (final_tmp.cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        print(f"Dequant {name} only PSNR: {psnr(orig, final_tmp)}")

if __name__ == "__main__":
    test_quant2()
