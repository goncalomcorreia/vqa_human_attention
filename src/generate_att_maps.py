#!/usr/bin/env python
import sys
sys.path.append('/home/s1670404/imageqa-san/src/')

from optimization_weight import *
from san_att_conv_twolayer_theano import *
from data_provision_att_vqa import *
from data_processing_vqa import *

import pickle
f = open('/home/s1670404/imageqa-san/data_vqa/answer_dict.pkl', 'r')
answer_dict = pickle.load(f)
f.close()
answer_dict = {v: k for k, v in answer_dict.iteritems()}

result = OrderedDict()

options, params, shared_params = load_model(
    '/home/s1670404/imageqa-san/expt/imageqa_best_0.523.model')

image_feat, input_idx, input_mask, label, \
dropout, cost, accu, pred_label, \
prob_attention_1, prob_attention_2 = build_model(
    shared_params, options)
