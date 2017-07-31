#!/usr/bin/env python

import skimage.transform
from skimage import io
import os
import numpy as np
import pickle as pkl
import h5py
from scipy.ndimage.filters import gaussian_filter

data_path = '/Users/goncalocorreia/vqa_human_attention/data_att_maps'

train_path = os.path.join(data_path, 'vqahat_train')
val_path = os.path.join(data_path, 'vqahat_val')

train_att_maps = []
att_map_qids = []

qids_id_tuple_list = [(img.split('_')[0],img.split('_')[1].split('.')[0]) for img in os.listdir(train_path)]

qids_ids_dict = {}

for tup in qids_id_tuple_list:
    if tup[0] in qids_ids_dict.keys():
        qids_ids_dict[tup[0]].append(tup[1])
    else:
        qids_ids_dict[tup[0]] = [tup[1]]

for key, value in qids_ids_dict.iteritems():
    qid = key
    qid_maps = []
    for qid_id in value:
        train_att_img = qid+'_'+qid_id+'.png'
        file_path = os.path.join(train_path, train_att_img)
        sample = io.imread(file_path)
        resized_sample = skimage.transform.resize(sample, (448,448), mode='reflect')
        downscaled_sample = skimage.transform.downscale_local_mean(resized_sample, factors=(32,32))
        flat_sample = downscaled_sample.flatten()

        if np.allclose(flat_sample, flat_sample[0]):
            continue

        normalized_sample = flat_sample/flat_sample.sum()
        qid_maps.append(normalized_sample)

    if len(qid_maps)<1:
        print qid
        continue

    qid_maps = np.asarray(qid_maps)
    flat_sample = qid_maps.mean(axis=0)
    normalized_sample = flat_sample/flat_sample.sum()
    train_att_maps.append(normalized_sample)
    att_map_qids.append(qid)

att_map_qids = [int(elem) for elem in att_map_qids]
with open('train.pkl', 'w') as f:
    pkl.dump(att_map_qids, f)

val_att_maps = []
val_att_map_qids = []
val_map_ids = []

for val_att_img in os.listdir(val_path):
    if val_att_img.split('_')[1].split('.')[0]!='1':
        continue
    qid = val_att_img.split('_')[0]
    map_id = val_att_img.split('_')[1].split('.')[0]
    file_path = os.path.join(val_path, val_att_img)
    sample = io.imread(file_path)
    resized_sample = skimage.transform.resize(sample, (448,448), mode='reflect')
    downscaled_sample=skimage.transform.pyramid_reduce(resized_sample, downscale=32)
    flat_sample = downscaled_sample.flatten()
    if flat_sample.sum()==0:
        continue
    else:
        normalized_sample = flat_sample/flat_sample.sum()
    if np.allclose(normalized_sample, normalized_sample[0]):
        continue
    val_att_maps.append(normalized_sample)
    val_att_map_qids.append(qid)
    val_map_ids.append(map_id)

val_att_map_qids = [int(elem) for elem in val_att_map_qids]
with open('val.pkl', 'w') as f:
    pkl.dump(val_att_map_qids, f)

map_qids = att_map_qids + val_map_ids
map_qids = [int(float(i)) for i in map_qids]
map_dist = np.zeros((len(map_qids), 196), dtype='float32')
map_dist[0:len(att_map_qids)] = train_att_maps
map_dist[len(att_map_qids):] = val_att_maps
map_dist_h5 = h5py.File('map_dist_196.h5', 'w')
map_dist_h5['label'] = map_dist
map_dist_h5.close()
