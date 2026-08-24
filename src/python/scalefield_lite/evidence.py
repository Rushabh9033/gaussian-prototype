import json
import uuid
import datetime
import subprocess
import torch
import sys
import hashlib
from typing import Dict, Any

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def get_gpu_info():
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)

def record_evidence(
    output_path: str,
    args: list,
    source_sha: str,
    model_id: str,
    model_revision: str,
    levels_data: list,
    artifacts: list,
    error: str = None
):
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
    
    import diffusers, transformers, accelerate
    
    evidence = {
        "run_uuid": str(uuid.uuid4()),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "command_arguments": args,
        "source_sha256": source_sha,
        "model_id": model_id,
        "model_revision": model_revision,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__
        },
        "hardware": {
            "gpu": get_gpu_info(),
            "peak_allocated_vram_gb": peak_allocated,
            "peak_reserved_vram_gb": peak_reserved
        },
        "levels": levels_data,
        "artifacts": artifacts,
        "error": error
    }
    
    with open(output_path, "w") as f:
        f.write(json.dumps(evidence) + "\n")
