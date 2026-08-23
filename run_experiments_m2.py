import subprocess
import sys

def run_experiment(name, count, steps, strategy):
    print(f"--- Running {name} ({strategy}, {count}) ---")
    out_zip = f"{name}.zip"
    
    # Encode
    subprocess.run([sys.executable, "main.py", "encode", "-i", r"C:\Users\RUSHABH\Downloads\porche.png", "-o", out_zip, "--count", str(count), "--steps", str(steps), "--max-dim", "512", "--strategy", strategy], check=True)
    
    # Benchmark
    res = subprocess.run([sys.executable, "main.py", "benchmark", "-i", out_zip], capture_output=True, text=True)
    print(res.stdout)
    
    # Info
    res_info = subprocess.run([sys.executable, "main.py", "info", "-i", out_zip], capture_output=True, text=True)
    print(res_info.stdout)
    
    # Render final PNG
    subprocess.run([sys.executable, "main.py", "render", "-i", out_zip, "-o", f"{name}.png", "--scale", "1.0"], check=True)

if __name__ == "__main__":
    run_experiment("porche_5k_adapt", 5000, 150, "adaptive")
    run_experiment("porche_10k_adapt", 10000, 150, "adaptive")
    
    # extract crops
    subprocess.run([sys.executable, "extract_crops.py"])
