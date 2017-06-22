#!/usr/bin/env python
import sys
sys.path.append('/home/s1670404/vqa_human_attention/src/')
sys.path.append('/home/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/home/s1670404/vqa_human_attention/src/models/')
from optimization_weight import *
from san_att_conv_twolayer_theano import *
from data_provision_att_vqa import *
from data_processing_vqa import *
import json
import pickle
f = open('/home/s1670404/vqa_human_attention/data_vqa/answer_dict.pkl', 'r')
answer_dict = pickle.load(f)
f.close()
answer_dict = {v: k for k, v in answer_dict.iteritems()}

import sys

model_path = sys.argv[1]
result_file_name = sys.argv[2]

result = OrderedDict()

options, params, shared_params = load_model(model_path)

image_feat, input_idx, input_mask, label, \
dropout, cost, accu, pred_label, \
prob_attention_1, prob_attention_2 = build_model(
    shared_params, options)

f_pass = theano.function(
    inputs=[
        image_feat,
        input_idx,
        input_mask],
    outputs=[pred_label],
    on_unused_input='warn')

data_provision_att_vqa = DataProvisionAttVqa(
    options['data_path'], options['feature_file'])

val_cost_list = []
val_accu_list = []
val_count = 0
dropout.set_value(numpy.float32(0.))

i=0

for batch_image_feat, batch_question, batch_answer_label in data_provision_att_vqa.iterate_batch(
        options['val_split'], options['batch_size']):
    input_idx, input_mask = process_batch(
        batch_question, reverse=options['reverse'])
    batch_image_feat = reshape_image_feat(batch_image_feat,
                                          options['num_region'],
                                          options['region_dim'])
    [pred_label] = f_pass(
        batch_image_feat,
        np.transpose(input_idx),
        np.transpose(input_mask))
    res =[]
    for pred in pred_label:
        ans = answer_dict[pred]
        ques_id = data_provision_att_vqa._question_id['val2'][i]
        i += 1
        result[ques_id] = ans

results = [result]


d = results[0]
res = []
for key, value in d.iteritems():
    res.append({'answer': value, 'question_id': int(key)})

with open('/home/s1670404/VQA/Results/'+result_file_name, 'w') as outfile:
    json.dump(res, outfile)
