import cv2
import numpy as np

im1 = cv2.imread('original/bad_joke_eel.png')
im2 = cv2.imread('20230920_121746/20230920_134727_sea_lion.png')

print(np.array_equal(im1, im2))

im_diff = im1.astype(int) - im2.astype(int)

print(im_diff.max())
# 142

print(im_diff.min())
# -101

im_diff_abs = np.abs(im_diff)

cv2.imwrite('diff1.png', im_diff_abs)