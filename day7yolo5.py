import torch
import cv2

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
dir = './yolov5-master/data/images/' # it's the master folder
imgs = [dir + f for f in ('zidane.jpg', 'snapey.jpg', 'catbullyingdog.jpg', 'minions.jpg')]  # batch of images ; works on snapey!:)

# Inference
results = model(imgs)
results.print()  # or .show(), .save()

# hello = cv2.imread('./yolov5-master/data/images/snapey.jpg', 1)
# cv2.imshow("hello", hello)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
