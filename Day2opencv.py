import cv2
import numpy as np
import matplotlib
img = cv2.imread("color_0.jpg",1)
flipped = cv2.flip(img, 90)
cv2.waitKey(10000)
cv2.destroyAllWindows()