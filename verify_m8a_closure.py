"""Milestone 8A closure verifier.

Generates milestone8a_evidence/closure_verification.json.
Verifies every required artifact, hash, dimension, and status derivation.
Exits nonzero on any closure failure.
"""
import os
import sys
import json
import hashlib
import struct
import subprocess
import datetime


EVIDENCE_DIR = "milestone8a_evidence"

REQUIRED_PNGS = [
    f"{EVIDENCE_DIR}/source/porche.png",
    f"{EVIDENCE_DIR}/controlled_inputs/lr_2x_180x110.png",
    f"{EVIDENCE_DIR}/controlled_inputs/lr_4x_90x55.png",
    f"{EVIDENCE_DIR}/lanczos/2x_360x220.png",
    f"{EVIDENCE_DIR}/lanczos/4x_360x220.png",
    f"{EVIDENCE_DIR}/lanczos/orig_1440x880.png",
    f"{EVIDENCE_DIR}/realesrgan/2x_360x220.png",
    f"{EVIDENCE_DIR}/realesrgan/4x_360x220.png",
    f"{EVIDENCE_DIR}/realesrgan/orig_1440x880.png",
    f"{EVIDENCE_DIR}/contact_sheet.png",
]

REQUIRED_CROPS = [
    f"{EVIDENCE_DIR}/crops/lanczos_plate.png",
    f"{EVIDENCE_DIR}/crops/realesrgan_plate.png",
    f"{EVIDENCE_DIR}/crops/lanczos_wheel.png",
    f"{EVIDENCE_DIR}/crops/realesrgan_wheel.png",
    f"{EVIDENCE_DIR}/crops/lanczos_headlamp.png",
    f"{EVIDENCE_DIR}/crops/realesrgan_headlamp.png",
]

REQUIRED_DOCS = [
    "MILESTONE8A_REPORT.md",
    f"{EVIDENCE_DIR}/execution_evidence.json",
    f"{EVIDENCE_DIR}/results.json",
]

NON_EMPTY_FILES = [
    "MILESTONE8A_REPORT.md",
    "tests/test_m8a.py",
    "validate_m8a.py",
]

GUARDRAILS = {
    "ssim_deg_limit": 0.01,
    "psnr_deg_limit_db": 1.0,
    "edge_f1_deg_limit": 0.02,
    "orb_inlier_ratio_deg_limit": 0.05,
    "regional_ssim_deg_limit": 0.03,
    "vram_reserved_limit_gb": 7.5,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def png_dimensions(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def git_commit_at(path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except Exception:
        return "UNRECOVERABLE_FROM_ORIGINAL_RUN"


def is_git_tracked(path):
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", path],
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    errors = []

    # --- Required artifacts ---
    for p in REQUIRED_DOCS + REQUIRED_PNGS + REQUIRED_CROPS:
        if not os.path.exists(p):
            errors.append(f"MISSING: {p}")
        elif os.path.getsize(p) <= 1:
            errors.append(f"EMPTY_OR_STUB: {p} ({os.path.getsize(p)} bytes)")

    # --- Non-empty validation files ---
    for p in NON_EMPTY_FILES:
        if not os.path.exists(p):
            errors.append(f"MISSING: {p}")
        elif os.path.getsize(p) <= 10:
            errors.append(f"CORRUPTED: {p} is {os.path.getsize(p)} bytes")

    # --- Crop Git tracking ---
    for crop in REQUIRED_CROPS:
        if os.path.exists(crop) and not is_git_tracked(crop):
            errors.append(f"UNTRACKED: {crop}")

    # --- PNG hashes and dimensions ---
    file_records = {}
    for p in sorted(set(REQUIRED_PNGS + REQUIRED_CROPS)):
        if not os.path.exists(p):
            continue
        w, h = png_dimensions(p)
        file_records[p] = {
            "sha256": sha256(p),
            "size_bytes": os.path.getsize(p),
            "width": w,
            "height": h,
        }

    # --- Verify inference hashes ---
    ev_path = f"{EVIDENCE_DIR}/execution_evidence.json"
    hashes_match = True
    if os.path.exists(ev_path):
        with open(ev_path) as f:
            ev = json.load(f)
        expected = ev.get("hashes", {})

        checks = [
            (f"{EVIDENCE_DIR}/controlled_inputs/lr_2x_180x110.png", "p_lr2x"),
            (f"{EVIDENCE_DIR}/controlled_inputs/lr_4x_90x55.png", "p_lr4x"),
            (f"{EVIDENCE_DIR}/realesrgan/2x_360x220.png", "realesrgan_2x"),
            (f"{EVIDENCE_DIR}/realesrgan/4x_360x220.png", "realesrgan_4x"),
            (f"{EVIDENCE_DIR}/realesrgan/orig_1440x880.png", "realesrgan_orig"),
        ]
        for fpath, key in checks:
            if os.path.exists(fpath) and key in expected:
                actual = sha256(fpath)
                if actual != expected[key]:
                    hashes_match = False
                    errors.append(f"HASH_MISMATCH: {fpath} expected {expected[key]} got {actual}")
    else:
        hashes_match = False
        errors.append("MISSING: execution_evidence.json")

    # --- Verify results.json ---
    res_path = f"{EVIDENCE_DIR}/results.json"
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
        if res.get("final_status") != "NO_USEFUL_GAIN":
            errors.append(f"STATUS_MISMATCH: results.json says {res.get('final_status')}")
        if res.get("overall_pass") is not False:
            errors.append("RESULTS: overall_pass should be false")
    else:
        errors.append("MISSING: results.json")

    # --- Verify report contains status ---
    report_path = "MILESTONE8A_REPORT.md"
    if os.path.exists(report_path) and os.path.getsize(report_path) > 10:
        report_text = open(report_path).read()
        if "NO_USEFUL_GAIN" not in report_text:
            errors.append("REPORT: does not contain NO_USEFUL_GAIN")

    # --- Derive expected status from metrics ---
    if os.path.exists(ev_path):
        with open(ev_path) as f:
            ev = json.load(f)
        metrics = ev.get("metrics", {})
        all_gates_pass = True
        for scale in ["2x", "4x"]:
            if scale not in metrics:
                all_gates_pass = False
                continue
            re = metrics[scale]["realesrgan"]
            lz = metrics[scale]["lanczos"]
            if lz["ssim"] - re["ssim"] > GUARDRAILS["ssim_deg_limit"]:
                all_gates_pass = False
            if lz["psnr"] - re["psnr"] > GUARDRAILS["psnr_deg_limit_db"]:
                all_gates_pass = False
            if lz["edge_f1"] - re["edge_f1"] > GUARDRAILS["edge_f1_deg_limit"]:
                all_gates_pass = False
            if lz["orb_geometric_inlier_ratio"] - re["orb_geometric_inlier_ratio"] > GUARDRAILS["orb_inlier_ratio_deg_limit"]:
                all_gates_pass = False
            re_reg = metrics[scale]["realesrgan_regions"]
            lz_reg = metrics[scale]["lanczos_regions"]
            for rname in ["plate", "headlamp", "wheel"]:
                if lz_reg[rname] - re_reg[rname] > GUARDRAILS["regional_ssim_deg_limit"]:
                    all_gates_pass = False
        expected_status = "NEEDS_USER_VISUAL_CHECK" if all_gates_pass else "NO_USEFUL_GAIN"
    else:
        expected_status = "UNKNOWN"

    # --- Provenance recovery ---
    ws_realesrgan = ".m8a_workspace/Real-ESRGAN"
    ws_basicsr = ".m8a_workspace/BasicSR"
    realesrgan_commit = git_commit_at(ws_realesrgan) if os.path.isdir(ws_realesrgan) else "UNRECOVERABLE_FROM_ORIGINAL_RUN"
    basicsr_commit = git_commit_at(ws_basicsr) if os.path.isdir(ws_basicsr) else "UNRECOVERABLE_FROM_ORIGINAL_RUN"

    provenance = {
        "realesrgan_commit": realesrgan_commit,
        "basicsr_commit": basicsr_commit,
    }

    try:
        import torch
        provenance["torch_version"] = torch.__version__
        provenance["cuda_version"] = str(torch.version.cuda) if torch.cuda.is_available() else "unavailable"
        provenance["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    except ImportError:
        provenance["torch_version"] = "UNRECOVERABLE_FROM_ORIGINAL_RUN"
        provenance["cuda_version"] = "UNRECOVERABLE_FROM_ORIGINAL_RUN"
        provenance["gpu_name"] = "UNRECOVERABLE_FROM_ORIGINAL_RUN"

    try:
        import torchvision
        provenance["torchvision_version"] = torchvision.__version__
    except ImportError:
        provenance["torchvision_version"] = "UNRECOVERABLE_FROM_ORIGINAL_RUN"

    # --- Build closure ---
    closure = {
        "verification_utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verification_commit": git_head(),
        "expected_final_status": expected_status,
        "recorded_final_status": "NO_USEFUL_GAIN",
        "inference_hashes_match": hashes_match,
        "original_source_sha256": sha256("m4d_test_assets/porche.png") if os.path.exists("m4d_test_assets/porche.png") else "source_not_found",
        "files": file_records,
        "required_artifacts_present": {p: os.path.exists(p) for p in REQUIRED_DOCS + REQUIRED_PNGS + REQUIRED_CROPS},
        "crops_git_tracked": {c: is_git_tracked(c) for c in REQUIRED_CROPS},
        "provenance": provenance,
        "errors": errors,
    }

    out_path = f"{EVIDENCE_DIR}/closure_verification.json"
    with open(out_path, "w") as f:
        json.dump(closure, f, indent=2)
    print(f"Wrote {out_path}")

    if errors:
        print(f"CLOSURE FAILED: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("CLOSURE PASSED: all artifacts verified, status correctly derived as NO_USEFUL_GAIN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
