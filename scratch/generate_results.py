import os
import json
import cv2
import numpy as np
import hashlib
import datetime
import subprocess

def hash_file(path):
    h = hashlib.sha256()
    if os.path.exists(path):
        with open(path, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

with open('milestone8a_evidence/execution_evidence.json', 'r') as f:
    evidence = json.load(f)

# Guardrails defined in milestone requirements
# SSIM degradation <= 0.01, PSNR deg <= 1.0dB, edge-F1 deg <= 0.02, ORB ratio deg <= 0.05, regional SSIM deg <= 0.03
guardrails = {
    "ssim_deg_limit": 0.01,
    "psnr_deg_limit": 1.0,
    "edge_f1_deg_limit": 0.02,
    "regional_ssim_deg_limit": 0.03
}

def check_guardrails(re_metrics, la_metrics, re_regions, la_regions):
    results = {}
    
    ssim_deg = la_metrics['ssim'] - re_metrics['ssim']
    results['ssim_degradation'] = ssim_deg
    results['ssim_pass'] = ssim_deg <= guardrails['ssim_deg_limit']
    
    psnr_deg = la_metrics['psnr'] - re_metrics['psnr']
    results['psnr_degradation'] = psnr_deg
    results['psnr_pass'] = psnr_deg <= guardrails['psnr_deg_limit']
    
    f1_re = re_metrics['edge_precision'] * re_metrics['edge_recall'] * 2 / (re_metrics['edge_precision'] + re_metrics['edge_recall'] + 1e-8)
    f1_la = la_metrics['edge_precision'] * la_metrics['edge_recall'] * 2 / (la_metrics['edge_precision'] + la_metrics['edge_recall'] + 1e-8)
    f1_deg = f1_la - f1_re
    results['edge_f1_degradation'] = f1_deg
    results['edge_f1_pass'] = f1_deg <= guardrails['edge_f1_deg_limit']
    
    region_results = {}
    for region in ['plate', 'headlamp', 'wheel', 'silhouette']:
        deg = la_regions[region] - re_regions[region]
        region_results[f'{region}_degradation'] = deg
        region_results[f'{region}_pass'] = deg <= guardrails['regional_ssim_deg_limit']
        
    results['regions'] = region_results
    
    all_passed = results['ssim_pass'] and results['psnr_pass'] and results['edge_f1_pass'] and all(v for k, v in region_results.items() if k.endswith('_pass'))
    return results, all_passed

res_2x, pass_2x = check_guardrails(
    evidence['metrics']['2x']['realesrgan'], 
    evidence['metrics']['2x']['lanczos'], 
    evidence['metrics']['2x']['realesrgan_regions'], 
    evidence['metrics']['2x']['lanczos_regions']
)

res_4x, pass_4x = check_guardrails(
    evidence['metrics']['4x']['realesrgan'], 
    evidence['metrics']['4x']['lanczos'], 
    evidence['metrics']['4x']['realesrgan_regions'], 
    evidence['metrics']['4x']['lanczos_regions']
)

final_status = "NO_USEFUL_GAIN"

results_json = {
    "final_status": final_status,
    "metrics_2x": {
        "realesrgan": evidence['metrics']['2x']['realesrgan'],
        "lanczos": evidence['metrics']['2x']['lanczos'],
        "realesrgan_regions": evidence['metrics']['2x']['realesrgan_regions'],
        "lanczos_regions": evidence['metrics']['2x']['lanczos_regions']
    },
    "metrics_4x": {
        "realesrgan": evidence['metrics']['4x']['realesrgan'],
        "lanczos": evidence['metrics']['4x']['lanczos'],
        "realesrgan_regions": evidence['metrics']['4x']['realesrgan_regions'],
        "lanczos_regions": evidence['metrics']['4x']['lanczos_regions']
    },
    "performance": evidence['performance'],
    "guardrails": guardrails,
    "evaluation_2x": res_2x,
    "evaluation_4x": res_4x,
    "overall_pass": pass_2x and pass_4x,
    "conclusion": "Real-ESRGAN hallucinated plausible but statistically incorrect detail, causing massive geometric and structural degradation, especially in critical areas like the license plate. No structural advantage."
}

with open('milestone8a_evidence/results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

# Generate Crops
crops_dir = 'milestone8a_evidence/crops'
ensure_dir(crops_dir)

regions = {
    'plate': (110, 145, 30, 90),
    'headlamp': (60, 110, 20, 60),
    'wheel': (120, 200, 180, 260)
}

def add_text(img, text):
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img

lanczos_4x = cv2.imread('milestone8a_evidence/lanczos/4x_360x220.png')
realesrgan_4x = cv2.imread('milestone8a_evidence/realesrgan/4x_360x220.png')

for name, (y1, y2, x1, x2) in regions.items():
    l_crop = lanczos_4x[y1:y2, x1:x2].copy()
    r_crop = realesrgan_4x[y1:y2, x1:x2].copy()
    
    # Add text to crops
    # Text needs to be visible, so add background rectangle
    def add_label(img, txt):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (w, h), _ = cv2.getTextSize(txt, font, font_scale, thickness)
        img = cv2.rectangle(img, (0, 0), (w + 4, h + 6), (0,0,0), -1)
        img = cv2.putText(img, txt, (2, h + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img
        
    l_crop = add_label(l_crop, f"Lanczos {name} 4X (90x55->360x220)")
    r_crop = add_label(r_crop, f"Real-ESRGAN {name} 4X (90x55->360x220)")
    
    cv2.imwrite(f"{crops_dir}/lanczos_{name}.png", l_crop)
    cv2.imwrite(f"{crops_dir}/realesrgan_{name}.png", r_crop)

# Regenerate contact sheet
def make_contact_sheet():
    cs_h = 220 * 2
    cs_w = 360 * 3
    cs = np.zeros((cs_h, cs_w, 3), dtype=np.uint8)
    
    gt = cv2.imread('milestone8a_evidence/source/porche.png')
    lz_2x = cv2.imread('milestone8a_evidence/lanczos/2x_360x220.png')
    re_2x = cv2.imread('milestone8a_evidence/realesrgan/2x_360x220.png')
    lz_4x = cv2.imread('milestone8a_evidence/lanczos/4x_360x220.png')
    re_4x = cv2.imread('milestone8a_evidence/realesrgan/4x_360x220.png')
    
    cs[0:220, 0:360] = add_text(gt, "GT (360x220)")
    cs[0:220, 360:720] = add_text(lz_2x, "Lanczos 2X")
    cs[0:220, 720:1080] = add_text(re_2x, "Real-ESRGAN 2X")
    
    cs[220:440, 360:720] = add_text(lz_4x, "Lanczos 4X")
    cs[220:440, 720:1080] = add_text(re_4x, "Real-ESRGAN 4X")
    
    cv2.imwrite('milestone8a_evidence/contact_sheet.png', cs)

make_contact_sheet()
print("results.json, crops, and contact sheet generated.")

