import json
import sys
import os
import torch

def preflight_check():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"Detected VRAM: {vram_gb:.2f} GB")
    
    if vram_gb < 24.0:
        print("Candidate A (Chain-of-Zoom) is BLOCKED: Requires 24GB VRAM.")
        
    print("Candidate B is BLOCKED: No reproducible implementation available.")
    print("Candidate C is BLOCKED: Insufficient VRAM to load VLM + SD + ControlNet locally.")

def main():
    preflight_check()
    print("All candidates blocked by hardware or licensing. Saving BLOCKED status.")
    
    # Save the status
    status = {
      "candidates": {
        "A": {"name": "Chain-of-Zoom", "status": "BLOCKED", "reason": "Hardware limitations (requires 24GB)"},
        "B": {"name": "Generative Powers of Ten", "status": "BLOCKED_NO_REPRODUCIBLE_IMPLEMENTATION", "reason": "No official implementation"},
        "C": {"name": "Direct-from-Root Structure-Conditioned", "status": "BLOCKED", "reason": "Insufficient VRAM for VLM + ControlNet"}
      },
      "overall_status": "BLOCKED"
    }
    
    with open("m4b_results.json", "w") as f:
        json.dump(status, f, indent=2)

if __name__ == "__main__":
    main()
