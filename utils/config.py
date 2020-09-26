from os.path import join

import numpy as np

scale = 32
seed = 12345
epochs = 100
batch_size = 32
max_boxes = 150
image_size = 416
data_dir = join('..', 'Dataset', 'VOC2012')
image_dir = 'IMAGES'
label_dir = 'LABELS'
classes = {'aeroplane': 0,
           'bicycle': 1,
           'bird': 2,
           'boat': 3,
           'bottle': 4,
           'bus': 5,
           'car': 6,
           'cat': 7,
           'chair': 8,
           'cow': 9,
           'diningtable': 10,
           'dog': 11,
           'horse': 12,
           'motorbike': 13,
           'person': 14,
           'pottedplant': 15,
           'sheep': 16,
           'sofa': 17,
           'train': 18,
           'tvmonitor': 19}
anchors = np.array([1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                    8.09892, 9.47112, 4.84053, 11.2364, 10.0071], np.float32)
