#!/usr/bin/env python
import sys
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/models/')
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')
from optimization_weight import *
from semi_joint_hsan_deepfix_att_theano import *
from data_provision_att_vqa_with_maps import *
from data_processing_vqa import *
import json
import pickle
# f = open('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/data_vqa/answer_dict.pkl', 'r')
# answer_dict = pickle.load(f)
# f.close()
# answer_dict = {v: k for k, v in answer_dict.iteritems()}
import numpy as np
import sys
from scipy.stats import spearmanr, pearsonr, sem

def correlation_coefficient(y_true, y_pred):

    num = np.sum(y_true * y_pred)
    den = np.sqrt((np.sum(y_true**2))*(np.sum(y_pred**2)))

    return num/den

model_path = sys.argv[1]

result = OrderedDict()

options, params, shared_params = load_model(model_path)
options['saliency_dropout'] = 0.5

image_feat, input_idx, input_mask, label, \
dropout, cost, accu, pred_label, \
prob_attention_1, prob_attention_2, map_cost, map_label = build_model(
    shared_params, params, options)

options['data_path'] = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/data_vqa/'
options['map_data_path'] = '/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/data_att_maps'


f_pass = theano.function(inputs = [image_feat, input_idx, input_mask],
                        outputs = [prob_attention_2],
                        on_unused_input='warn')

data_provision_att_vqa = DataProvisionAttVqaWithMaps(options['data_path'],
                                                     options['feature_file'],
                                                     options['map_data_path'])

dropout.set_value(numpy.float32(0.))

res = np.array([])

for batch_image_feat, batch_question, batch_answer_label, batch_map_label in data_provision_att_vqa.iterate_batch(
        'val1', options['batch_size']):

    input_idx, input_mask = process_batch(
        batch_question, reverse=options['reverse'])
    batch_image_feat = reshape_image_feat(batch_image_feat,
                                          options['num_region'],
                                          options['region_dim'])

    [prob_attention_2] = f_pass(batch_image_feat, np.transpose(input_idx),
                         np.transpose(input_mask))

    # cross_ent = -np.sum(np.log(prob_attention_1)*batch_map_label, axis=0)
    # res = np.append(res, np.mean(cross_ent))
    for aa,bb in zip(batch_map_label, prob_attention_2):
        if np.isnan(spearmanr(aa,bb)[0]):
            import pdb; pdb.set_trace()
        res = np.append(res, 2*spearmanr(aa,bb)[0])

print "Correlation of validation: "+str(np.mean(res))+" , SEM: "+str(sem(res))
