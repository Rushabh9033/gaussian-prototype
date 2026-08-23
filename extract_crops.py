from PIL import Image
import os

def extract_crops(img_path, prefix):
    if not os.path.exists(img_path):
        print(f"Not found: {img_path}")
        return
    
    img = Image.open(img_path)
    
    # 360x220 Porsche image crops
    # Adjust coordinates based on where the Porsche is.
    # Assuming it's a standard car photo
    crops = {
        "plate": (160, 160, 200, 180),
        "headlight": (220, 120, 260, 150),
        "wheel": (80, 150, 130, 200),
        "edge": (250, 80, 300, 130),
        "foliage": (20, 20, 70, 70)
    }
    
    for name, box in crops.items():
        crop = img.crop(box)
        crop.save(f"{prefix}_{name}.png")
        print(f"Saved {prefix}_{name}.png")

def main():
    images = [
        ("C:\\Users\\RUSHABH\\Downloads\\porche.png", "orig"),
        ("porche_5k.png", "m1_5k"),
        ("porche_10k.png", "m1_10k"),
        ("porche_5k_adapt.png", "m2_5k"),
        ("porche_10k_adapt.png", "m2_10k")
    ]
    for path, prefix in images:
        extract_crops(path, prefix)

if __name__ == "__main__":
    main()
