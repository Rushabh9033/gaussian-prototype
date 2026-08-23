import subprocess
import json

def run_experiment(name, count, steps):
    print(f"--- Running {name} ---")
    out_zip = f"{name}.zip"
    
    # Encode
    subprocess.run(["python", "main.py", "encode", "-i", r"C:\Users\RUSHABH\Downloads\porche.png", "-o", out_zip, "--count", str(count), "--steps", str(steps), "--max-dim", "512"])
    
    # Benchmark
    res = subprocess.run(["python", "main.py", "benchmark", "-i", out_zip], capture_output=True, text=True)
    print(res.stdout)
    
    # Info
    res_info = subprocess.run(["python", "main.py", "info", "-i", out_zip], capture_output=True, text=True)
    print(res_info.stdout)
    
    # Render final PNG
    subprocess.run(["python", "main.py", "render", "-i", out_zip, "-o", f"{name}.png", "--scale", "1.0"])

run_experiment("porche_5k", 5000, 150)
run_experiment("porche_10k", 10000, 150)
