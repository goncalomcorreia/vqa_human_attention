#!/usr/bin/env python

import pdb
import os
import numpy as np
import cv2

import caffe

def load_model(prototxt_file, model_file, image_size, mean):
    caffe.set_mode_gpu()
    model = dict()
    model['net'] = caffe.Net(prototxt_file, model_file, caffe.TEST)
    model['image_size'] = image_size
    model['mean'] = mean
    return model

def get_feature(image, model, blob_name):
    image = image.astype(np.float32, copy=True)
    image -= model['mean']
    image = np.transpose(image, axes = (2, 0, 1))
    image = image[np.newaxis, :, :, :]
    model['net'].forward(data=image.astype(np.float32, copy=False))
    feature = model['net'].blobs[blob_name].data.copy()
    return feature



# with open(image_list_file) as f:
#     for line in f:
#         image_list.append(line.split()[0])
#         image = cv2.imread(os.path.join(image_folder, image_list[-1]))
#         border = (image.shape[0] - image_size) / 2
#         image = image[border : border + image_size,
#                       border : border + image_size,
#                       :]
#         feat = get_feature(image, model, 'fc7')
#         break
