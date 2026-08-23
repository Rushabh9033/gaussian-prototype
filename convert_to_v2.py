import argparse
from src.python.package import read_package, create_package

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    
    manifest, data, metrics = read_package(args.input)
    W = manifest["encoded_width"]
    H = manifest["encoded_height"]
    
    pos = data[:, 0:2]
    scale = data[:, 2:4]
    rot = data[:, 4]
    color = data[:, 5:8]
    opacity = data[:, 8]
    
    # We pass the same metrics so it has the same sha256 and creation time
    create_package(args.output, pos, scale, rot, color, opacity, W, H, metrics, format_version=2.0)

if __name__ == "__main__":
    main()
