import argparse
import time
import os
import hashlib
from PIL import Image
import torch
import numpy as np
import datetime
from .encoder import encode_image, GaussianModel
from .renderer import render_image
from .package import create_package, read_package

# Optional imports for metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    psnr, ssim = None, None

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def encode_cmd(args):
    device = get_device()
    print(f"Using device: {device}")
    
    img = Image.open(args.input).convert('RGB')
    orig_W, orig_H = img.size
    
    # Scale down if needed
    W, H = orig_W, orig_H
    if max(W, H) > args.max_dim:
        scale = args.max_dim / max(W, H)
        W = int(W * scale)
        H = int(H * scale)
        img = img.resize((W, H), Image.LANCZOS)
        
    img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_t = img_t.to(device)
    
    print(f"Encoding {W}x{H} with {args.count} Gaussians for {args.steps} steps using {args.strategy} strategy...")
    
    model, encode_time = encode_image(
        img_t, 
        num_gaussians=args.count, 
        steps=args.steps, 
        device=device,
        seed=args.seed,
        strategy=args.strategy
    )
    
    pos, scale, rot, color, opacity = model.get_params()
    
    # Render final image for preview and metrics AT ORIGINAL RESOLUTION
    with torch.no_grad():
        render_scale = orig_W / W if W > 0 else 1.0
        pos_eval = pos * render_scale
        scale_eval = scale * render_scale
        final_img_t = render_image(orig_W, orig_H, pos_eval, scale_eval, rot, color, opacity)
        final_img_np = (final_img_t.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        preview_img = Image.fromarray(final_img_np)
        
    # Metrics
    orig_img_np = np.array(Image.open(args.input).convert('RGB'))
    
    metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": hash_file(args.input),
        "original_width": orig_W,
        "original_height": orig_H,
        "encoded_width": W,
        "encoded_height": H,
        "encode_time_seconds": encode_time,
        "device_used": str(device)
    }
    
    if psnr and ssim:
        metrics["psnr"] = float(psnr(orig_img_np, final_img_np))
        win_size = min(7, min(orig_W, orig_H))
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(3, win_size)
        metrics["ssim"] = float(ssim(orig_img_np, final_img_np, channel_axis=-1, data_range=255, win_size=win_size))
        
    # Save package
    create_package(
        args.output,
        pos.detach().cpu().numpy(),
        scale.detach().cpu().numpy(),
        rot.detach().cpu().numpy(),
        color.detach().cpu().numpy(),
        opacity.detach().cpu().numpy(),
        W, H, metrics, preview_img, seed=args.seed
    )
    
    pkg_size = os.path.getsize(args.output)
    print(f"Package saved to {args.output} ({pkg_size / 1024:.1f} KB)")
    print(metrics)

def render_cmd(args):
    device = get_device()
    manifest, data, metrics = read_package(args.input)
    
    W = int(manifest["encoded_width"] * args.scale)
    H = int(manifest["encoded_height"] * args.scale)
    
    data_t = torch.from_numpy(data).to(device)
    pos = data_t[:, 0:2] * args.scale
    scale = data_t[:, 2:4] * args.scale
    rot = data_t[:, 4]
    color = data_t[:, 5:8]
    opacity = data_t[:, 8]
    
    start = time.time()
    with torch.no_grad():
        final_img_t = render_image(W, H, pos, scale, rot, color, opacity)
    render_time = time.time() - start
    
    final_img_np = (final_img_t.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(final_img_np).save(args.output)
    print(f"Rendered {W}x{H} in {render_time:.3f}s, saved to {args.output}")

def info_cmd(args):
    manifest, data, metrics = read_package(args.input)
    print("--- Manifest ---")
    for k, v in manifest.items():
        print(f"{k}: {v}")
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
def benchmark_cmd(args):
    device = get_device()
    manifest, data, metrics = read_package(args.input)
    
    data_t = torch.from_numpy(data).to(device)
    rot = data_t[:, 4]
    color = data_t[:, 5:8]
    opacity = data_t[:, 8]
    
    print(f"Benchmarking {args.input} on {device}")
    for scale in [1.0, 2.0, 4.0]:
        W = int(manifest["encoded_width"] * scale)
        H = int(manifest["encoded_height"] * scale)
        
        pos = data_t[:, 0:2] * scale
        scale_param = data_t[:, 2:4] * scale
        
        # warmup
        with torch.no_grad():
            _ = render_image(W, H, pos, scale_param, rot, color, opacity)
            
        start = time.time()
        iters = 5
        with torch.no_grad():
            for _ in range(iters):
                _ = render_image(W, H, pos, scale_param, rot, color, opacity)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        avg_time = (time.time() - start) / iters
        print(f"Scale {scale}x ({W}x{H}): {avg_time:.4f}s / frame ({1/avg_time:.1f} FPS)")

def main():
    parser = argparse.ArgumentParser(description="2D Gaussian Prototype CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Encode
    enc_parser = subparsers.add_parser("encode")
    enc_parser.add_argument("-i", "--input", required=True, help="Input image path")
    enc_parser.add_argument("-o", "--output", required=True, help="Output zip path")
    enc_parser.add_argument("--count", type=int, default=5000, help="Number of Gaussians")
    enc_parser.add_argument("--steps", type=int, default=500, help="Optimization steps")
    enc_parser.add_argument("--max-dim", type=int, default=256, help="Max image dimension during encode")
    enc_parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    enc_parser.add_argument("--strategy", type=str, choices=['random', 'adaptive'], default='random', help="Initialization strategy")
    
    # Render
    ren_parser = subparsers.add_parser("render")
    ren_parser.add_argument("-i", "--input", required=True, help="Input zip path")
    ren_parser.add_argument("-o", "--output", required=True, help="Output png path")
    ren_parser.add_argument("--scale", type=float, default=1.0, help="Output resolution scale")
    
    # Info
    info_parser = subparsers.add_parser("info")
    info_parser.add_argument("-i", "--input", required=True, help="Input zip path")
    
    # Benchmark
    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("-i", "--input", required=True, help="Input zip path")
    
    args = parser.parse_args()
    
    if args.command == "encode":
        encode_cmd(args)
    elif args.command == "render":
        render_cmd(args)
    elif args.command == "info":
        info_cmd(args)
    elif args.command == "benchmark":
        benchmark_cmd(args)

if __name__ == "__main__":
    main()
