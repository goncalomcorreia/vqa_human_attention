#!/usr/bin/env python

import skimage.transform
from skimage import io
import os
import numpy as np
import pickle as pkl
import h5py
from scipy.ndimage.filters import gaussian_filter

data_path = '/Users/goncalocorreia/VQA/maps'

train_path = os.path.join(data_path, 'train')
val_path = os.path.join(data_path, 'val')

train_att_maps = []
att_map_imids = []

for train_att_img in os.listdir(train_path):
    imid = train_att_img.split('_')[-1].split('.')[0]
    file_path = os.path.join(train_path, train_att_img)
    sample = io.imread(file_path)
    resized_sample = skimage.transform.resize(sample, (448,448), mode='reflect')
    downscaled_sample = skimage.transform.downscale_local_mean(resized_sample, factors=(32,32))
    flat_sample = downscaled_sample.flatten()
    if flat_sample.sum()==0:
        normalized_sample = flat_sample
    else:
        normalized_sample = flat_sample/flat_sample.sum()
    train_att_maps.append(normalized_sample)
    att_map_imids.append(imid)

att_map_imids = [int(elem) for elem in att_map_imids]
with open('train_salicon.pkl', 'w') as f:
    pkl.dump(att_map_imids, f)

val_att_maps = []
val_att_map_imids = []

for val_att_img in os.listdir(val_path):
    imid = val_att_img.split('_')[-1].split('.')[0]
    file_path = os.path.join(val_path, val_att_img)
    sample = io.imread(file_path)
    resized_sample = skimage.transform.resize(sample, (448,448), mode='reflect')
    downscaled_sample=skimage.transform.pyramid_reduce(resized_sample, downscale=32)
    flat_sample = downscaled_sample.flatten()
    normalized_sample = flat_sample/flat_sample.sum()
    val_att_maps.append(normalized_sample)
    val_att_map_imids.append(imid)

val_att_map_imids = [int(elem) for elem in val_att_map_imids]
with open('val_salicon.pkl', 'w') as f:
    pkl.dump(val_att_map_imids, f)

map_imids = att_map_imids + val_att_map_imids
map_imids = [int(float(i)) for i in map_imids]
map_dist = np.zeros((len(map_imids), 196), dtype='float32')
map_dist[0:len(att_map_imids)] = train_att_maps
map_dist[len(att_map_imids):] = val_att_maps
map_dist_h5 = h5py.File('map_dist_196_salicon.h5', 'w')
map_dist_h5['label'] = map_dist
map_dist_h5.close()
