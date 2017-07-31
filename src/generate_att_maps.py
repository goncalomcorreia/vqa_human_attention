#!/usr/bin/env python
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')
import sys
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/data-providers/')
sys.path.append('/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/src/models/')
from optimization_weight import *
from data_provision_att_vqa_with_maps import *
from data_processing_vqa import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io

model_path = sys.argv[1]
model_script = sys.argv[2]
val_split = sys.argv[3]
folder_name = sys.argv[4]

# model_path = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/expt/hsan_deepfix/hsan_deepfix_lmda_0.2_smallkernels_best_0.536.model'
# model_path = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/expt/baseline/baseline_best_0.523.model'

if model_script == 'baseline':
    from san_att_conv_twolayer_theano import *
    options, params, shared_params = load_model(model_path)
    image_feat, input_idx, input_mask, label, \
    dropout, cost, accu, pred_label, \
    prob_attention_1, prob_attention_2 = build_model(
        shared_params, options)
elif model_script=='hsan':
    from semi_joint_hsan_deepfix_att_theano import *
    options, params, shared_params = load_model(model_path)
    options['saliency_dropout'] = 0.5
    image_feat, input_idx, input_mask, \
    label, dropout, ans_cost, accu, pred_label, \
    prob_attention_1, prob_attention_2, map_cost, map_label = build_model(shared_params, params, options)

save_folder = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/'+folder_name+'/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

get_att_2 = theano.function(
    inputs=[
        image_feat,
        input_idx,
        input_mask],
    outputs=[prob_attention_2],
    on_unused_input='warn')

data_path = '/afs/inf.ed.ac.uk/group/synproc/Goncalo/data_vqa'

options['data_path']=data_path
options['val_split']=val_split
data_provision_att_vqa = DataProvisionAttVqaWithMaps(
    options['data_path'], options['feature_file'], '/afs/inf.ed.ac.uk/user/s16/s1670404/vqa_human_attention/data_att_maps')

dropout.set_value(numpy.float32(0.))

i=0

for batch_image_feat, batch_question, batch_answer_label, batch_map_label in data_provision_att_vqa.iterate_batch(
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

    for att_map in prob_attention_2:
        # alpha_img = skimage.transform.pyramid_expand(att_map.reshape(14,14), upscale=16, sigma=20)
        alpha_img = att_map.reshape(14,14)
        name = str(data_provision_att_vqa._question_id[options['val_split']][i]) + '.png'
        plt.imsave(save_folder+name, alpha_img, cmap=cm.Greys_r)
        i+=1
