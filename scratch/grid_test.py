import cv2
import os

img = cv2.imread('m4d_test_assets/porche.png')
os.makedirs('scratch/grid', exist_ok=True)

h, w, _ = img.shape
grid_h = h // 4
grid_w = w // 4

for i in range(4):
    for j in range(4):
        y1, y2 = i * grid_h, (i+1) * grid_h
        x1, x2 = j * grid_w, (j+1) * grid_w
        cv2.imwrite(f'scratch/grid/r{i}_c{j}.png', img[y1:y2, x1:x2])
