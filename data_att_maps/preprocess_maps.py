#!/usr/bin/env python

import skimage.transform
from skimage import io
import os
import numpy as np
import pickle as pkl

data_path = '/Users/goncalocorreia/vqa_human_attention/data_att_maps'

train_path = os.path.join(data_path, 'vqahat_train')
val_path = os.path.join(data_path, 'vqahat_val')

train_att_maps = []
att_map_qids = []

for train_att_img in os.listdir(train_path):
    qid = train_att_img.split('_')[0]
    file_path = os.path.join(train_path, train_att_img)
    sample = io.imread(file_path)
    resized_sample = skimage.transform.resize(sample, (448,448), mode='reflect')
    downscaled_sample=skimage.transform.pyramid_reduce(resized_sample, downscale=32)
    flat_sample = downscaled_sample.flatten()
    normalized_sample = flat_sample/flat_sample.sum()
    train_att_maps.append(normalized_sample)
    att_map_qids.append(qid)

with open('train_att_maps.pkl', 'w') as f:
    pkl.dump(np.array(train_att_maps), f)
    pkl.dump(att_map_qids, f)

val_att_maps = []
val_att_map_qids = []
val_map_ids = []

for val_att_img in os.listdir(val_path):
    qid = val_att_img.split('_')[0]
    map_id = val_att_img.split('_')[1].split('.')[0]
    file_path = os.path.join(val_path, val_att_img)
    sample = io.imread(file_path)
    resized_sample = skimage.transform.resize(sample, (448,448), mode='reflect')
    downscaled_sample=skimage.transform.pyramid_reduce(resized_sample, downscale=32)
    flat_sample = downscaled_sample.flatten()
    normalized_sample = flat_sample/flat_sample.sum()
    val_att_maps.append(normalized_sample)
    val_att_map_qids.append(qid)
    val_map_ids.append(map_id)

with open('val_att_maps.pkl', 'w') as f:
    pkl.dump(np.array(val_att_maps), f)
    pkl.dump(val_att_map_qids, f)
    pkl.dump(val_map_ids, f)
