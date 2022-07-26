from telnetlib import X3PAD
import torch
import cv2
import plot_utils as pu
import numpy as np
from glob import glob
import os
# import plot.utils 
# import plot_results



# HI IF UR READING THIS WE WERE EDITING THE NAMES OF QIFAN'S HAND DATASET CUZ THEY WERE LIKE '_color000000005.jpg)
data_array = np.load('./handsdataset/datasetuno/center.npy')
index = np.load('./handsdataset/datasetuno/index.npy')

def rename():  # he thought the naming was ugly so he basically got rid of the 'color' and 'jpg' part, cut off the excess 0s, and turned it into a string
    new_output = './handsdataset/datasetuno/new'
    os.makedirs(new_output, exist_ok=True)

    for file in glob('./handsdataset/datasetuno/pics/color*.jpg'):
        num = ""
        for c in file:
            if c.isdigit():
                num = num + c
        num = int(num)
        filename = os.path.join(new_output, f'{num}.jpg')
        image = cv2.imread(file)
        cv2.imwrite(filename, image)

# print("\nOG Data:\n", data_array)
new_output = './handsdataset/datasetuno/new'  # creating the new directory cuz i don't want to mess with the old one
for idx, id in enumerate(index):  
    with open(f'./handsdataset/datasetuno/new/{id}.txt', 'w+') as label:  
        # he moved the saved files into a new folder and created a txt file for each annotated file with its x,y,width,height
        # the txt files only appeared for some of the og data files bc only some of them were actually annotated 
        # also the og data files are basically a bunch of frames per second and a lot of them were the exact same pic lol
        center_x = data_array[idx][0] 
        center_y = data_array[idx][1]
        filename = os.path.join(new_output, f'{id}.jpg')
        image = cv2.imread(filename)
        height, width, _ = image.shape
        norm_center_x = center_x / width  # normalizing this puts the x and y coordiantes btwn 0 and 1
        norm_center_y = center_y / height
        norm_height = 90.0 / height
        norm_width = 90.0 / width
        print(f'0 {norm_center_x} {norm_center_y} {norm_width} {norm_height}', file=label, end='')
# basically yolo's data input requires a few things: class_id center_x center_y width height
# https://roboflow.com/formats/yolov5-pytorch-txt
# https://stackoverflow.com/questions/66563034/does-yolov5-pytorch-use-a-different-labeling-format



# Each datafile notes the coordinates of the top left, top right, bottom right, bottom left - clockwise



# # Model
# model1 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Images
# # dir = './yolov5-master/data/images/coco128/images/train2017/' # COCO128
# testdir = './yolov5-master/data/i/'
# imgs = [testdir + f for f in ('guy-surfing.jpg','bookcat.jpg')]  # batch of images ; works only on 2+ images


# # Inference
# results = model1(imgs)
# results.print()  # or .show(), .save()

# # plot_results('path/to/results.csv')  # plot 'results.csv' as 'results.png'
