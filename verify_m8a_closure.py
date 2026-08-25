import os
import json
import hashlib
import datetime
import subprocess

def hash_file(path):
    h = hashlib.sha256()
    if os.path.exists(path):
        with open(path, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()

def get_git_commit(path='.'):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    except Exception:
        return "UNRECOVERABLE_FROM_ORIGINAL_RUN"

def get_file_stats(path):
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    import struct
    # Simple PNG dimension parsing
    with open(path, 'rb') as f:
        head = f.read(24)
        if len(head) == 24 and head[:8] == b'\x89PNG\r\n\x1a\n':
            w, h = struct.unpack('>II', head[16:24])
            mode = "RGB" # Simplified
            return {"size_bytes": size, "width": w, "height": h, "mode": mode}
    return {"size_bytes": size}

evidence_dir = 'milestone8a_evidence'
pngs = []
for root, _, files in os.walk(evidence_dir):
    for f in files:
        if f.endswith('.png'):
            pngs.append(os.path.join(root, f).replace('\\', '/'))

file_records = {}
for p in sorted(pngs):
    st = get_file_stats(p)
    file_records[p] = {
        "sha256": hash_file(p),
        **st
    }

original_source_hash = hash_file('m4d_test_assets/porche.png')

with open(f'{evidence_dir}/execution_evidence.json', 'r') as f:
    exec_ev = json.load(f)

hashes_match = (
    hash_file(f'{evidence_dir}/controlled_inputs/lr_4x_90x55.png') == exec_ev['hashes']['p_lr4x'] and
    hash_file(f'{evidence_dir}/realesrgan/2x_360x220.png') == exec_ev['hashes']['realesrgan_2x'] and
    hash_file(f'{evidence_dir}/realesrgan/4x_360x220.png') == exec_ev['hashes']['realesrgan_4x'] and
    hash_file(f'{evidence_dir}/realesrgan/orig_1440x880.png') == exec_ev['hashes']['realesrgan_orig']
)

required_artifacts = [
    'MILESTONE8A_REPORT.md',
    f'{evidence_dir}/execution_evidence.json',
    f'{evidence_dir}/results.json',
    f'{evidence_dir}/contact_sheet.png'
]
artifacts_present = {f: os.path.exists(f) for f in required_artifacts}

workspace_realesrgan = '.m8a_workspace/Real-ESRGAN'
workspace_basicsr = '.m8a_workspace/BasicSR'

realesrgan_commit = get_git_commit(workspace_realesrgan) if os.path.exists(workspace_realesrgan) else "UNRECOVERABLE_FROM_ORIGINAL_RUN"
basicsr_commit = get_git_commit(workspace_basicsr) if os.path.exists(workspace_basicsr) else "UNRECOVERABLE_FROM_ORIGINAL_RUN"

import torch
import torchvision
cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

closure = {
    "verification_utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "verification_commit": get_git_commit('.'),
    "files": file_records,
    "original_source_sha256": original_source_hash,
    "inference_hashes_match": hashes_match,
    "required_artifacts_present": artifacts_present,
    "final_status": "NO_USEFUL_GAIN",
    "provenance": {
        "realesrgan_commit": realesrgan_commit,
        "basicsr_commit": basicsr_commit,
        "python_packages": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__
        },
        "hardware": {
            "cuda_version": cuda_ver,
            "gpu_name": gpu_name
        }
    }
}

with open(f'{evidence_dir}/closure_verification.json', 'w') as f:
    json.dump(closure, f, indent=2)

print("closure_verification.json generated.")
