import cv2
import numpy as np
import matplotlib

img = cv2.imread("color_14.jpg",1)
height, width = img.shape[0:2]
croppedImage = img[441:621, 656:836]
cv2.imshow("color_14.jpg", img)
cv2.imshow("Cropped Image", croppedImage)
status = cv2.imwrite("/home/irvl/Documents/Centered_image_14.png",croppedImage)
print("Image written to file system: ",status)
cv2.waitKey(10000)
cv2.destroyAllWindows()
