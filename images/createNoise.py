import numpy as np
import cv2

# 画像の幅と高さ
width, height = 224, 224

# 画像をランダムな画素値で初期化
random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

# 画像を表示（必要に応じて）
cv2.imshow('Random Image', random_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像をファイルに保存（必要に応じて）
cv2.imwrite('random_image.png', random_image)
