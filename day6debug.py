import cv2


testimg = cv2.imread('./training_set/dogs/dog1.jpg', 0)
print(testimg)
cv2.imshow("testimg", testimg)
cv2.waitKey(10000)
cv2.destroyAllWindows