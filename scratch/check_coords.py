import cv2
img = cv2.imread('m4d_test_assets/porche.png')

# y_start:y_end, x_start:x_end
plate = img[110:145, 30:90]
headlamp = img[60:110, 20:60]
wheel = img[120:200, 180:260]

cv2.imwrite('scratch/crops/plate_exact.png', plate)
cv2.imwrite('scratch/crops/headlamp_exact.png', headlamp)
cv2.imwrite('scratch/crops/wheel_exact.png', wheel)
