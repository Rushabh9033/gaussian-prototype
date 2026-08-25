import cv2
import os

img = cv2.imread('m4d_test_assets/porche.png')
os.makedirs('scratch/crops', exist_ok=True)

# Guesses (y, y+h, x, x+w)
regions = {
    'plate': (150, 180, 150, 210),
    'wheel': (130, 200, 40, 110),
    'headlamp': (110, 140, 60, 100),
    'silhouette': (50, 200, 30, 330)
}

for name, (y1, y2, x1, x2) in regions.items():
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f'scratch/crops/{name}.png', crop)
