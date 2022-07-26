import os
import torch
import cv2
import pandas
import numpy as np
# 'custom', path='/reallybigyolo/yolov5-',source='local' /yolov5-master/runs/train/exp33/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5-master/runs/train/exp33/weights/best.pt')
cap = cv2.VideoCapture(0)
# while( cap.isOpened() ) :
#     ret,img = cap.read()
#     cv2.imshow("lll",img)
#     k = cv2.waitKey(10)
#     if k == 27:
#         break
axes = None
NUM_FRAMES = 200 # you can change this
for i in range(NUM_FRAMES):
    # Load frame from the camera
    ret, frame = cap.read()

    # Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

    # Run frame through network
    class_IDs, scores, bounding_boxes = net(rgb_nd)

    # Display the result
    img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
    gcv.utils.viz.cv_plot_image(img)
    cv2.waitKey(1)
results = model(cap)
print(results)


# dir = './roboflowv18/test/images'
# for image in os.listdir(dir):
#     test = os.path.join(dir,image)
#     print(test)

#     newimage=cv2.imread(str(test))
    
    # results = model(newimage)
    # # output=classes[np.argmax(results)]
# print("The predicted class is ", results)
    


