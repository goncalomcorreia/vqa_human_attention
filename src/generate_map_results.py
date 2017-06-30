#!/usr/bin/env python
import sys
sys.path.append('/home/s1670404/vqa_human_attention/src/')
sys.path.append('/home/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/home/s1670404/vqa_human_attention/src/models/')
from optimization_weight import *
from san_att_conv_twolayer_theano import *
from data_provision_att_vqa_with_maps import *
from data_processing_vqa import *
import json
import pickle
f = open('/home/s1670404/vqa_human_attention/data_vqa/answer_dict.pkl', 'r')
answer_dict = pickle.load(f)
f.close()
answer_dict = {v: k for k, v in answer_dict.iteritems()}
import numpy as np
import sys

model_path = sys.argv[1]
result_file_name = sys.argv[2]

result = OrderedDict()

options, params, shared_params = load_model(model_path)

image_feat, input_idx, input_mask, label, \
dropout, cost, accu, pred_label, \
prob_attention_1, prob_attention_2 = build_model(
    shared_params, options)

options['map_data_path'] = '/home/s1670404/vqa_human_attention/data_att_maps'

f_pass = theano.function(inputs = [image_feat, input_idx, input_mask],
                        outputs = [prob_attention_1],
                        on_unused_input='warn')

data_provision_att_vqa = DataProvisionAttVqaWithMaps(options['data_path'],
                                                     options['feature_file'],
                                                     options['map_data_path'])

dropout.set_value(numpy.float32(0.))

res = np.array([])

for batch_image_feat, batch_question, batch_answer_label, batch_map_label in data_provision_att_vqa.iterate_batch(
        options['val_split'], options['batch_size']):

    input_idx, input_mask = process_batch(
        batch_question, reverse=options['reverse'])
    batch_image_feat = reshape_image_feat(batch_image_feat,
                                          options['num_region'],
                                          options['region_dim'])

    [prob_attention_1] = f_pass(batch_image_feat, np.transpose(input_idx),
                         np.transpose(input_mask))

    cross_ent = -np.sum(np.log(prob_attention_1)*batch_map_label, axis=0)
    res = np.append(res, np.mean(cross_ent))

print "Cross Entropy of validation: "+str(np.mean(res))
