import cv2
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle

img = cv2.imread("color_14.jpg",1)
height, width = img.shape[0:2]
start=(656,621)
end=(836,441)
color=(0,255,0)
thickness=2
img=cv2.rectangle(img,start,end,color,thickness)  # draw a bounding box
img=cv2.circle(img,(746,531),radius=5,color=(0,0,255),thickness=-5)
cv2.imshow("color_14.jpg", img)
cv2.waitKey(10000)
cv2.destroyAllWindows()