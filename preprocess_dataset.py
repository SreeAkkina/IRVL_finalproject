import os
dataset_path = 'reallybigyolo/handsdataset/datasetuno/'

image_path = os.path.join(dataset_path, 'images/*')
label_path = os.path.join(dataset_path, 'labels/*')

from glob import glob

image_files = glob(image_path)
label_files = glob(label_path)

for file in image_files:
    basename = os.path.basename(file)
    filename = basename.split('.')[0]
    in_label = False

    for label in label_files:
        label_base = os.path.basename(label)
        label_filename = label_base.split('.')[0]
        if label_filename == filename:
            in_label = True

    if not in_label:
        os.remove(file)
        print(file)