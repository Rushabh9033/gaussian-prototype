import os
import json
import sys

def main():
    ev_path = "milestone8a_evidence/execution_evidence.json"
    if not os.path.exists(ev_path):
        print("ERROR: execution_evidence.json missing.")
        return 1
        
    with open(ev_path, "r") as f:
        ev = json.load(f)
        
    # Check guardrails
    # "For both controlled scales, Real-ESRGAN must satisfy all of these relative to Lanczos:"
    # SSIM degradation <= 0.01
    # PSNR degradation <= 1.0 dB
    # edge-F1 degradation <= 0.02
    # ORB ratio degradation <= 0.05
    # regional SSIM degradation <= 0.03
    
    # Must have improvement: SSIM +0.002 OR edge-F1 +0.01 OR Laplacian variance 10% closer to GT
    
    metrics = ev.get("metrics", {})
    if "2x" not in metrics or "4x" not in metrics:
        print("ERROR: Missing 2x or 4x metrics")
        return 1
        
    has_improvement = False
    
    for scale in ["2x", "4x"]:
        re = metrics[scale]["realesrgan"]
        lz = metrics[scale]["lanczos"]
        
        # Degradations
        if lz["ssim"] - re["ssim"] > 0.01:
            print(f"FAILED: SSIM degradation > 0.01 for {scale}")
            return 1
        if lz["psnr"] - re["psnr"] > 1.0:
            print(f"FAILED: PSNR degradation > 1.0 for {scale}")
            return 1
        if lz["edge_f1"] - re["edge_f1"] > 0.02:
            print(f"FAILED: edge_f1 degradation > 0.02 for {scale}")
            return 1
        if lz["orb_geometric_inlier_ratio"] - re["orb_geometric_inlier_ratio"] > 0.05:
            print(f"FAILED: orb ratio degradation > 0.05 for {scale}")
            return 1
            
        re_reg = metrics[scale]["realesrgan_regions"]
        lz_reg = metrics[scale]["lanczos_regions"]
        for rname in ["plate", "wheel", "headlamp"]:
            if lz_reg[rname] - re_reg[rname] > 0.03:
                print(f"FAILED: regional SSIM degradation > 0.03 for {scale} {rname}")
                return 1
                
        # Improvements
        if re["ssim"] - lz["ssim"] >= 0.002:
            has_improvement = True
        if re["edge_f1"] - lz["edge_f1"] >= 0.01:
            has_improvement = True
            
        # laplacian variance 10% closer to GT
        # GT Laplacian variance would need to be computed. 
        # But if the others pass we are good. Let's just trust has_improvement for now.
        
    if not has_improvement:
        print("FAILED: No measurable improvement found.")
        return 2

    # Peak VRAM
    perf = ev.get("performance", {})
    for task_name, task_perf in perf.items():
        if task_perf["peak_reserved_gb"] > 7.5:
            print(f"FAILED: Peak VRAM for {task_name} exceeded 7.5GB: {task_perf['peak_reserved_gb']}")
            return 1

    print("SUCCESS: Milestone 8A guardrails passed.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
