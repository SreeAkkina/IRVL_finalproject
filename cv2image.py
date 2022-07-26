import cv2

filename = 'color_0.jpg'

photo = cv2.imread(filename)

photo = cv2.flip(photo, 1)
cv2.imshow('title', photo)

cv2.waitKey(0)