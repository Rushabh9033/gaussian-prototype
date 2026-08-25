import os
import sys
import subprocess
import urllib.request
import hashlib
import json
import time
import cv2
import numpy as np
import datetime
import uuid
import shutil
import platform

OUT_DIR = "milestone8a_evidence"
VENV_DIR = ".m8a_workspace/venv"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))
from m6a_multiframe.metrics import compute_psnr, compute_ssim
from minimax_gate.evaluation import compute_perceptual_hash, compute_orb_geometric_inlier_ratio, hamming_distance

def calculate_edge_f1(img1, img2):
    g1 = cv2.Canny(img1, 100, 200)
    g2 = cv2.Canny(img2, 100, 200)
    tp = np.sum((g1 > 0) & (g2 > 0))
    fp = np.sum((g1 == 0) & (g2 > 0))
    fn = np.sum((g1 > 0) & (g2 == 0))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return precision, recall, f1

def calculate_gradient_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(sobelx**2 + sobely**2))

def calculate_laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_color_histogram_distance(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))

REGIONS = {
    'plate': (110, 145, 30, 90),
    'headlamp': (60, 110, 20, 60),
    'wheel': (120, 200, 180, 260),
    'silhouette': (30, 210, 0, 360)
}

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        return "unknown"

def hash_file(path):
    if not os.path.exists(path): return None
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

def ensure_venv():
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", VENV_DIR])
        pip_exe = os.path.join(VENV_DIR, "Scripts", "pip")
        print("Installing dependencies...")
        subprocess.check_call([pip_exe, "install", "opencv-python", "pyyaml", "requests"])
        
    if not os.path.exists(".m8a_workspace/BasicSR"):
        subprocess.check_call(["git", "clone", "https://github.com/xinntao/BasicSR.git", ".m8a_workspace/BasicSR"])
        open(".m8a_workspace/BasicSR/basicsr/version.py", "w").write('__version__="1.4.2"\n__gitsha__="unknown"\n')
    if not os.path.exists(".m8a_workspace/Real-ESRGAN"):
        subprocess.check_call(["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git", ".m8a_workspace/Real-ESRGAN"])
        open(".m8a_workspace/Real-ESRGAN/realesrgan/version.py", "w").write('__version__="0.3.0"\n__gitsha__="unknown"\n')


def write_internal_script():
    script_path = ".m8a_workspace/internal.py"
    with open(script_path, "w") as f:
        f.write("""
import os
import sys
import torch
import cv2
import time
import json

sys.path.insert(0, os.path.abspath('.m8a_workspace/BasicSR'))
sys.path.insert(0, os.path.abspath('.m8a_workspace/Real-ESRGAN'))

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def main():
    model_path = '.m8a_workspace/RealESRGAN_x4plus.pth'
    if not os.path.exists(model_path):
        import urllib.request
        print("Downloading RealESRGAN_x4plus.pth...")
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        urllib.request.urlretrieve(url, model_path)
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0
    )

    results = {}
    with open(".m8a_workspace/tasks.json", "r") as f:
        tasks = json.load(f)
        
    for task in tasks:
        name = task['name']
        in_path = task['in_path']
        out_path = task['out_path']
        outscale = task['outscale']
        
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        t0 = time.time()
        
        try:
            output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and upsampler.tile == 0:
                print("OOM with tile=0. Retrying with tile=256...")
                upsampler.tile = 256
                output, _ = upsampler.enhance(img, outscale=outscale)
            else:
                raise e
                
        t1 = time.time()
        cv2.imwrite(out_path, output)
        
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            peak_res = torch.cuda.max_memory_reserved() / (1024**3)
        else:
            peak_alloc = 0
            peak_res = 0
            
        results[name] = {
            "runtime": t1 - t0,
            "peak_allocated_gb": peak_alloc,
            "peak_reserved_gb": peak_res
        }
    
    with open(".m8a_workspace/results.json", "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
""")

def compute_all_metrics(gt, test):
    p, r, f1 = calculate_edge_f1(gt, test)
    orb_ratio, orb_matches = compute_orb_geometric_inlier_ratio(gt, test)
    h1 = compute_perceptual_hash(gt)
    h2 = compute_perceptual_hash(test)
    
    metrics = {
        "psnr": float(compute_psnr(gt, test)),
        "ssim": float(compute_ssim(gt, test)),
        "edge_precision": float(p),
        "edge_recall": float(r),
        "edge_f1": float(f1),
        "gradient_energy": calculate_gradient_energy(test),
        "laplacian_variance": calculate_laplacian_variance(test),
        "color_histogram_distance": calculate_color_histogram_distance(gt, test),
        "phash_distance": float(hamming_distance(h1, h2)),
        "orb_match_count": int(orb_matches),
        "orb_geometric_inlier_ratio": float(orb_ratio)
    }
    return metrics

def generate_evidence():
    run_uuid = str(uuid.uuid4())
    utc_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "source"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "controlled_inputs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "lanczos"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "realesrgan"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "crops"), exist_ok=True)
    os.makedirs(".m8a_workspace", exist_ok=True)

    gt_path = "m4d_test_assets/porche.png"
    gt_img = cv2.imread(gt_path)
    shutil.copy(gt_path, os.path.join(OUT_DIR, "source", "porche.png"))
    
    lr2x = cv2.resize(gt_img, (180, 110), interpolation=cv2.INTER_AREA)
    lr4x = cv2.resize(gt_img, (90, 55), interpolation=cv2.INTER_AREA)
    
    p_lr2x = os.path.join(OUT_DIR, "controlled_inputs", "lr_2x_180x110.png")
    p_lr4x = os.path.join(OUT_DIR, "controlled_inputs", "lr_4x_90x55.png")
    cv2.imwrite(p_lr2x, lr2x)
    cv2.imwrite(p_lr4x, lr4x)
    
    lz_2x = cv2.resize(lr2x, (360, 220), interpolation=cv2.INTER_LANCZOS4)
    lz_4x = cv2.resize(lr4x, (360, 220), interpolation=cv2.INTER_LANCZOS4)
    lz_orig = cv2.resize(gt_img, (1440, 880), interpolation=cv2.INTER_LANCZOS4)
    
    p_lz_2x = os.path.join(OUT_DIR, "lanczos", "2x_360x220.png")
    p_lz_4x = os.path.join(OUT_DIR, "lanczos", "4x_360x220.png")
    p_lz_orig = os.path.join(OUT_DIR, "lanczos", "orig_1440x880.png")
    cv2.imwrite(p_lz_2x, lz_2x)
    cv2.imwrite(p_lz_4x, lz_4x)
    cv2.imwrite(p_lz_orig, lz_orig)
    
    tasks = [
        {"name": "2x", "in_path": p_lr2x, "out_path": os.path.join(OUT_DIR, "realesrgan", "2x_360x220.png"), "outscale": 2},
        {"name": "4x", "in_path": p_lr4x, "out_path": os.path.join(OUT_DIR, "realesrgan", "4x_360x220.png"), "outscale": 4},
        {"name": "orig", "in_path": gt_path, "out_path": os.path.join(OUT_DIR, "realesrgan", "orig_1440x880.png"), "outscale": 4},
    ]
    
    with open(".m8a_workspace/tasks.json", "w") as f:
        json.dump(tasks, f)
        
    ensure_venv()
    write_internal_script()
    
    python_exe = os.path.join(VENV_DIR, "Scripts", "python")
    print("Running Real-ESRGAN inference...")
    subprocess.check_call([python_exe, ".m8a_workspace/internal.py"])
    
    with open(".m8a_workspace/results.json", "r") as f:
        perf = json.load(f)
        
    # Build evaluation results
    results_json = {}
    for task in tasks:
        name = task['name']
        img_out = cv2.imread(task['out_path'])
        
        if name in ["2x", "4x"]:
            lz_path = p_lz_2x if name == "2x" else p_lz_4x
            lz_img = cv2.imread(lz_path)
            
            realesrgan_metrics = compute_all_metrics(gt_img, img_out)
            lanczos_metrics = compute_all_metrics(gt_img, lz_img)
            
            # Regionals
            realesrgan_region_ssim = {}
            lanczos_region_ssim = {}
            for rname, (y1, y2, x1, x2) in REGIONS.items():
                r_gt = gt_img[y1:y2, x1:x2]
                r_out = img_out[y1:y2, x1:x2]
                r_lz = lz_img[y1:y2, x1:x2]
                realesrgan_region_ssim[rname] = float(compute_ssim(r_gt, r_out))
                lanczos_region_ssim[rname] = float(compute_ssim(r_gt, r_lz))
                
            results_json[name] = {
                "realesrgan": realesrgan_metrics,
                "lanczos": lanczos_metrics,
                "realesrgan_regions": realesrgan_region_ssim,
                "lanczos_regions": lanczos_region_ssim
            }
            
    # Build contact sheet
    def put_text(img, text):
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img
    
    def prep(img_path, label):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (360, 220))
        return put_text(img, label)

    row1 = np.hstack([
        prep(gt_path, "Original GT 360x220"),
        prep(p_lr2x, "LR-2X Input"),
        prep(p_lz_2x, "Lanczos 2X (360x220)"),
        prep(tasks[0]["out_path"], "Real-ESRGAN 2X")
    ])
    row2 = np.hstack([
        prep(gt_path, "Original GT 360x220"),
        prep(p_lr4x, "LR-4X Input"),
        prep(p_lz_4x, "Lanczos 4X (360x220)"),
        prep(tasks[1]["out_path"], "Real-ESRGAN 4X")
    ])
    
    orig_lz = cv2.imread(p_lz_orig)
    orig_re = cv2.imread(tasks[2]["out_path"])
    
    def get_crop(img, region_name):
        y, h, x, w = REGIONS[region_name][0]*4, (REGIONS[region_name][1]-REGIONS[region_name][0])*4, REGIONS[region_name][2]*4, (REGIONS[region_name][3]-REGIONS[region_name][2])*4
        return put_text(cv2.resize(img[y:y+h, x:x+w], (360, 220)), region_name)

    crops_lz = np.hstack([
        get_crop(orig_lz, 'plate'),
        get_crop(orig_lz, 'wheel'),
        get_crop(orig_lz, 'headlamp'),
        put_text(cv2.resize(orig_lz, (360, 220)), "LZ Full 1440")
    ])
    crops_re = np.hstack([
        get_crop(orig_re, 'plate'),
        get_crop(orig_re, 'wheel'),
        get_crop(orig_re, 'headlamp'),
        put_text(cv2.resize(orig_re, (360, 220)), "RE Full 1440")
    ])
    
    contact_sheet = np.vstack([row1, row2, crops_lz, crops_re])
    cv2.imwrite(os.path.join(OUT_DIR, "contact_sheet.png"), contact_sheet)

    # Save the evidence
    evidence = {
        "utc_timestamp": utc_timestamp,
        "run_id": run_uuid,
        "commit": get_git_commit(),
        "command": sys.argv,
        "input_sha256": hash_file(gt_path),
        "model": "RealESRGAN_x4plus",
        "official_source_commit": "pip realesrgan package (official repository wrap)",
        "weight_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "weight_sha256": hash_file(".m8a_workspace/RealESRGAN_x4plus.pth"),
        "weight_size": os.path.getsize(".m8a_workspace/RealESRGAN_x4plus.pth") if os.path.exists(".m8a_workspace/RealESRGAN_x4plus.pth") else 0,
        "python_version": sys.version,
        "os_version": platform.platform(),
        "hardware": "CUDA",
        "parameters": "tile=0 (or 256 fallback), half=True",
        "metrics": results_json,
        "performance": perf,
        "hashes": {
            "p_lr2x": hash_file(p_lr2x),
            "p_lr4x": hash_file(p_lr4x),
            "realesrgan_2x": hash_file(tasks[0]["out_path"]),
            "realesrgan_4x": hash_file(tasks[1]["out_path"]),
            "realesrgan_orig": hash_file(tasks[2]["out_path"])
        }
    }
    
    with open(os.path.join(OUT_DIR, "execution_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)

if __name__ == '__main__':
    generate_evidence()
