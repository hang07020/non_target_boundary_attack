import numpy as np
import cv2

height, width = 224, 224
b, g, r = 255, 0, 0  # orange
image = np.zeros((height, width, 3), np.uint8)
image[:, :, 0] = b
image[:, :, 1] = g
image[:, :, 2] = r


cv2.imwrite('mean.png', image)