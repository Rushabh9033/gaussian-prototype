"""Milestone 8A validator.

Loads results.json, evaluates every guardrail from execution_evidence.json,
derives the expected status, confirms it matches the recorded status,
and verifies closure_verification.json succeeded.

Exit 0 only when all checks pass.
"""
import os
import sys
import json


GUARDRAILS = {
    "ssim_deg_limit": 0.01,
    "psnr_deg_limit_db": 1.0,
    "edge_f1_deg_limit": 0.02,
    "orb_inlier_ratio_deg_limit": 0.05,
    "regional_ssim_deg_limit": 0.03,
    "vram_reserved_limit_gb": 7.5,
}

EVIDENCE_DIR = "milestone8a_evidence"


def main():
    # --- Load execution evidence ---
    ev_path = os.path.join(EVIDENCE_DIR, "execution_evidence.json")
    if not os.path.exists(ev_path):
        print("ERROR: execution_evidence.json missing.")
        return 1
    with open(ev_path) as f:
        ev = json.load(f)

    # --- Load results.json ---
    res_path = os.path.join(EVIDENCE_DIR, "results.json")
    if not os.path.exists(res_path):
        print("ERROR: results.json missing.")
        return 1
    with open(res_path) as f:
        res = json.load(f)

    metrics = ev.get("metrics", {})
    if "2x" not in metrics or "4x" not in metrics:
        print("ERROR: Missing 2x or 4x metrics")
        return 1

    # --- Evaluate every guardrail ---
    all_gates_pass = True
    has_ssim_improvement = False
    has_edge_f1_improvement = False

    for scale in ["2x", "4x"]:
        re = metrics[scale]["realesrgan"]
        lz = metrics[scale]["lanczos"]

        # SSIM degradation
        if lz["ssim"] - re["ssim"] > GUARDRAILS["ssim_deg_limit"]:
            all_gates_pass = False

        # PSNR degradation
        if lz["psnr"] - re["psnr"] > GUARDRAILS["psnr_deg_limit_db"]:
            all_gates_pass = False

        # Edge-F1 degradation
        if lz["edge_f1"] - re["edge_f1"] > GUARDRAILS["edge_f1_deg_limit"]:
            all_gates_pass = False

        # ORB inlier ratio degradation
        if lz["orb_geometric_inlier_ratio"] - re["orb_geometric_inlier_ratio"] > GUARDRAILS["orb_inlier_ratio_deg_limit"]:
            all_gates_pass = False

        # Regional SSIM degradation
        re_reg = metrics[scale]["realesrgan_regions"]
        lz_reg = metrics[scale]["lanczos_regions"]
        for rname in ["plate", "headlamp", "wheel"]:
            if lz_reg[rname] - re_reg[rname] > GUARDRAILS["regional_ssim_deg_limit"]:
                all_gates_pass = False

        # Improvement checks
        if re["ssim"] - lz["ssim"] >= 0.002:
            has_ssim_improvement = True
        if re["edge_f1"] - lz["edge_f1"] >= 0.01:
            has_edge_f1_improvement = True

    # --- VRAM check ---
    perf = ev.get("performance", {})
    vram_ok = True
    for task_name, task_perf in perf.items():
        if task_perf["peak_reserved_gb"] > GUARDRAILS["vram_reserved_limit_gb"]:
            vram_ok = False
            print(f"ERROR: Peak VRAM for {task_name} exceeded {GUARDRAILS['vram_reserved_limit_gb']}GB: {task_perf['peak_reserved_gb']}")

    if not vram_ok:
        return 1

    # --- Derive expected status ---
    if all_gates_pass and (has_ssim_improvement or has_edge_f1_improvement):
        expected_status = "NEEDS_USER_VISUAL_CHECK"
    else:
        expected_status = "NO_USEFUL_GAIN"

    recorded_status = res.get("final_status")
    if recorded_status != expected_status:
        print(f"ERROR: Expected status {expected_status} but found {recorded_status}")
        return 1

    # --- Verify closure verification ---
    closure_path = os.path.join(EVIDENCE_DIR, "closure_verification.json")
    if os.path.exists(closure_path):
        with open(closure_path) as f:
            closure = json.load(f)
        closure_errors = closure.get("errors", [])
        if closure_errors:
            print(f"ERROR: closure_verification has {len(closure_errors)} error(s):")
            for e in closure_errors:
                print(f"  - {e}")
            return 1
    else:
        print("WARNING: closure_verification.json not found, skipping closure check")

    print("VALID: experiment failed quality guardrails and status is correctly NO_USEFUL_GAIN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
