import argparse
import subprocess
import os

def run_cmd(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True)
    parser.add_argument('-p', '--package', required=True)
    args = parser.parse_args()
    
    out_pkg = "porche_10k_progressive_v2.zip"
    
    # 1. Convert
    cmd = f"python convert_to_v2.py -i {args.package} -o {out_pkg}"
    run_cmd(cmd)
    
    # 2. Info
    cmd = f"python main.py info -i {out_pkg}"
    run_cmd(cmd)
    
    # 3. Evaluate Decoded
    cmd = f"python evaluate_v2.py -i {args.image} -p {out_pkg}"
    run_cmd(cmd)

if __name__ == "__main__":
    main()
