#!/usr/bin/env python
import sys
sys.path.append('/home/s1670404/vqa_human_attention/src/')

from optimization_weight import *
from san_att_conv_twolayer_theano import *
from data_provision_att_vqa import *
from data_processing_vqa import *

import pickle
f = open('/home/s1670404/vqa_human_attention/data_vqa/answer_dict.pkl', 'r')
answer_dict = pickle.load(f)
f.close()
answer_dict = {v: k for k, v in answer_dict.iteritems()}

result = OrderedDict()

options, params, shared_params = load_model(
    '/home/s1670404/vqa_human_attention/expt/imageqa_best_0.523.model')

image_feat, input_idx, input_mask, label, \
dropout, cost, accu, pred_label, \
prob_attention_1, prob_attention_2 = build_model(
    shared_params, options)


get_att_2 = theano.function(
    inputs=[
        image_feat,
        input_idx,
        input_mask],
    outputs=[prob_attention_2],
    on_unused_input='warn')

options['data_path']='/home/s1670404/vqa_human_attention/data_vqa'
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
    [prob_attention_2] = get_att_2(
        batch_image_feat,
        np.transpose(input_idx),
        np.transpose(input_mask))

    import pdb; pdb.set_trace()
