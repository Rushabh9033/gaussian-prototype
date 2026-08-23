import subprocess
import os
import json

def run_cmd(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    img_path = r"C:\Users\RUSHABH\Downloads\porche.png"
    out_pkg = "porche_10k_progressive_v2.zip"
    
    # 1. Encode
    cmd = f"python -u main.py encode -i {img_path} -o {out_pkg} --count 10000 --steps 150 --max-dim 512 --strategy adaptive --seed 42 --progressive"
    run_cmd(cmd)
    
    # 2. Benchmark Full
    cmd = f"python -u main.py benchmark -i {out_pkg} --layer all"
    run_cmd(cmd)
    
    # 3. Benchmark Base
    cmd = f"python -u main.py benchmark -i {out_pkg} --layer base"
    run_cmd(cmd)
    
    # 4. Info
    cmd = f"python -u main.py info -i {out_pkg}"
    run_cmd(cmd)

if __name__ == "__main__":
    main()
